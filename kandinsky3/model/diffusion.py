import math

import torch
from einops import rearrange
from tqdm import tqdm
import pdb

from .utils import get_tensor_items


def get_named_beta_schedule(schedule_name, timesteps):
    if schedule_name == "linear":
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(
            beta_start, beta_end, timesteps, dtype=torch.float32
        )
    elif schedule_name == "cosine":
        alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        betas = []
        for i in range(timesteps):
            t1 = i / timesteps
            t2 = (i + 1) / timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float32)


class BaseDiffusion:

    def __init__(self, betas, percentile=None, gen_noise=torch.randn_like):
        self.betas = betas
        self.num_timesteps = betas.shape[0]

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=betas.dtype), self.alphas_cumprod[:-1]])

        # calculate q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculate q(x_{t-1} | x_t, x_0)
        self.posterior_mean_coef_1 = torch.sqrt(self.alphas_cumprod_prev) * betas / (1. - self.alphas_cumprod)
        self.posterior_mean_coef_2 = torch.sqrt(alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(
            torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        )

        self.percentile = percentile
        self.time_scale = 1000 // self.num_timesteps
        self.gen_noise = gen_noise
        self.jump_length = 3

    def process_x_start(self, x_start):
        bs, ndims = x_start.shape[0], len(x_start.shape[1:])
        if self.percentile is not None:
            quantile = torch.quantile(
                rearrange(x_start, 'b ... -> b (...)').abs(),
                self.percentile,
                dim=-1
            )
            quantile = torch.clip(quantile, min=1.)
            quantile = quantile.reshape(bs, *((1,) * ndims))
            return torch.clip(x_start, -quantile, quantile) / quantile
        else:
            return torch.clip(x_start, -1., 1.)

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        if mask is not None:
            x_t = mask * x_start + (1. - mask) * x_t
        return x_t

    def inp_q_sample(self, x_t, t, l, noise=None):
        if noise is None:
            noise = self.gen_noise(x_t)

        res = get_tensor_items(self.alphas_cumprod, t + l, x_t.shape) / get_tensor_items(self.alphas_cumprod, t, x_t.shape)
        sqrt_alphas_cumprod = torch.sqrt(res)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - res + 1e-8)

        x_t_l = sqrt_alphas_cumprod * x_t + sqrt_one_minus_alphas_cumprod * noise
        return x_t_l

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean_coef_1 = get_tensor_items(self.posterior_mean_coef_1, t, x_start.shape)
        posterior_mean_coef_2 = get_tensor_items(self.posterior_mean_coef_2, t, x_t.shape)
        posterior_mean = posterior_mean_coef_1 * x_start + posterior_mean_coef_2 * x_t

        posterior_variance = get_tensor_items(self.posterior_variance, t, x_start.shape)
        posterior_log_variance = get_tensor_items(self.posterior_log_variance, t, x_start.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def txt_img_guidance(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0, 
                        mask=None, masked_latent=None, unconditional_mask=None, unconditional_latent=None):
        large_x = x.repeat(3, 1, 1, 1)
        large_t = t.repeat(3)
        
        uncondition_context = torch.zeros_like(context)
        uncondition_context_mask = torch.zeros_like(context_mask)
        uncondition_context[:, 0] = null_embedding
        uncondition_context_mask[:, 0] = 1


        large_context = torch.cat([context, uncondition_context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask, uncondition_context_mask])
        
        if mask is not None:
            mask = mask.repeat(2, 1, 1, 1)
            mask = torch.cat([mask, unconditional_mask])

        if masked_latent is not None:

            masked_latent = masked_latent.repeat(2, 1, 1, 1)

            masked_latent = torch.cat([masked_latent, unconditional_latent])

        if model.in_layer.in_channels == 5:
            large_x = torch.cat([large_x, mask], dim=1)
        
        elif model.in_layer.in_channels == 9:
            large_x = torch.cat([large_x, mask, masked_latent], dim=1)
        
        pred_large_noise = model(large_x, large_t * self.time_scale, large_context, large_context_mask.bool())

        pred_noise, pred_image_noise, uncond_pred_noise = torch.chunk(pred_large_noise, 3)

        # pdb.set_trace()

        pred_noise = uncond_pred_noise + guidance_weight_text * (pred_noise - pred_image_noise) + guidance_weight_image * (pred_image_noise - uncond_pred_noise)
        return pred_noise

    def text_guidance(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0, 
                        mask=None, masked_latent=None):
        large_x = x.repeat(2, 1, 1, 1)
        large_t = t.repeat(2)
        
        uncondition_context = torch.zeros_like(context)
        uncondition_context_mask = torch.zeros_like(context_mask)
        uncondition_context[:, 0] = null_embedding
        uncondition_context_mask[:, 0] = 1
        large_context = torch.cat([context, uncondition_context])
        large_context_mask = torch.cat([context_mask, uncondition_context_mask])
        
        if mask is not None:
            mask = mask.repeat(2, 1, 1, 1)
        if masked_latent is not None:
            masked_latent = masked_latent.repeat(2, 1, 1, 1)

        if model.in_layer.in_channels == 5:
            large_x = torch.cat([large_x, mask], dim=1)
        
        elif model.in_layer.in_channels == 9:
            large_x = torch.cat([large_x, mask, masked_latent], dim=1)
        
        pred_large_noise = model(large_x, large_t * self.time_scale, large_context, large_context_mask.bool())
        pred_noise, uncond_pred_noise = torch.chunk(pred_large_noise, 2)
        pred_noise = (guidance_weight_text + 1.) * pred_noise - guidance_weight_text * uncond_pred_noise
        return pred_noise

    def p_mean_variance(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0, 
                        mask=None, masked_latent=None, unconditional_mask=None, unconditional_latent=None):
        
        if guidance_weight_image == 1.0:
            pred_noise = self.text_guidance(model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image, mask, masked_latent)
        else:
            pred_noise = self.txt_img_guidance(model, x, t, context, context_mask, null_embedding, guidance_weight_text - 1, guidance_weight_image, 
                                                mask, masked_latent, unconditional_mask=unconditional_mask, unconditional_latent=unconditional_latent)

        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, pred_noise.shape)
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod * pred_noise) / sqrt_alphas_cumprod
        pred_x_start = self.process_x_start(pred_x_start)

        pred_mean, pred_var, pred_log_var = self.q_posterior_mean_variance(pred_x_start, x, t)
        return pred_mean, pred_var, pred_log_var

    def inp_p_mean_variance(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0, 
                            mask=None, masked_latent=None, unconditional_mask=None, unconditional_latent=None):
       
        if guidance_weight_image == 1.0:
            pred_noise = self.text_guidance(model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image, mask, masked_latent)
        else:
            pred_noise = self.txt_img_guidance(model, x, t, context, context_mask, null_embedding, guidance_weight_text - 1, guidance_weight_image, 
                                                mask, masked_latent, unconditional_mask=unconditional_mask, unconditional_latent=unconditional_latent)

        # coef = get_tensor_items(torch.log(1 + 2 * self.posterior_variance), t, mask.shape)
        # guidance_mask = mask * coef * guidance_weight
        # print(guidance_mask.shape)
        # print(pred_noise * guidance_mask)
        # pred_noise = (guidance_mask + 1.) * pred_noise - guidance_mask * uncond_pred_noise

        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, pred_noise.shape)
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod * pred_noise) / sqrt_alphas_cumprod
        pred_x_start = self.process_x_start(pred_x_start)

        pred_x_start = mask * masked_latent + (1. - mask) * pred_x_start

        pred_mean, pred_var, pred_log_var = self.q_posterior_mean_variance(pred_x_start, x, t)
        return pred_mean, pred_var, pred_log_var

    @torch.no_grad()
    def p_sample(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0, 
                mask=None, masked_latent=None, unconditional_mask=None, unconditional_latent=None):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        pred_mean, _, pred_log_var = self.p_mean_variance(model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=guidance_weight_image,
                                                          mask=mask, masked_latent=masked_latent, unconditional_mask=unconditional_mask, unconditional_latent=unconditional_latent)
        noise = torch.randn_like(x)
        mask = (t != 0).reshape(bs, *((1,) * ndims))
        sample = pred_mean + mask * torch.exp(0.5 * pred_log_var) * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, context, context_mask, null_embedding, guidance_weight_text, 
                        guidance_weight_image=1.0, mask=None, masked_latent=None, image_latent=None, vae=None, strength=1.0):
        
        if image_latent is None:
            img = torch.randn(*shape, device=device)
            t_start = self.num_timesteps
        else:
            init_timestep = min(int(self.num_timesteps * strength), self.num_timesteps)
            t_start = max(self.num_timesteps - init_timestep, 0)
            img = self.q_sample(image_latent, init_timestep)
        
        unconditional_mask = None
        unconditional_latent = None
        if mask is not None and masked_latent is not None:
            unconditional_mask = torch.zeros_like(mask)
            unconditional_latent = torch.nn.functional.interpolate(unconditional_mask, size=(masked_latent.shape[2] * 8, masked_latent.shape[3] * 8)).repeat_interleave(3, dim=1)
            unconditional_latent = vae.encode(unconditional_latent)

        time = list(range(t_start))[::-1]
        
        for time in tqdm(time, position=0):
            time = torch.tensor([time] * shape[0], device=device)
            img = self.p_sample(
                model, img, time, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image, 
                mask=mask, masked_latent=masked_latent, unconditional_mask=unconditional_mask, 
                unconditional_latent=unconditional_latent
            )
        return img

    @torch.no_grad()
    def inp_p_sample(self, model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=1.0,
                     mask=None, masked_latent=None, image_latent=None, unconditional_mask=None, unconditional_latent=None):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        pred_mean, _, pred_log_var = self.inp_p_mean_variance(model, x, t, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=guidance_weight_image, 
                                                              mask=mask, masked_latent=masked_latent, unconditional_mask=unconditional_mask, unconditional_latent=unconditional_latent)
        noise = torch.randn_like(x)
        mask = (t != 0).reshape(bs, *((1,) * ndims))
        sample = pred_mean + mask * torch.exp(0.5 * pred_log_var) * noise
        return sample

    @torch.no_grad()
    def inp_p_sample_loop(self, model, shape, device, context, context_mask, null_embedding, guidance_weight_text, 
                          guidance_weight_image=0.0, mask=None, masked_latent=None, image_latent=None, vae=None,  strength=1.0):
        
        if image_latent is None:
            img = torch.randn(*shape, device=device)
            t_start = self.num_timesteps
        else:
            init_timestep = min(int(self.num_timesteps * strength), self.num_timesteps)
            t_start = max(self.num_timesteps - init_timestep, 0)
            img = self.q_sample(image_latent, init_timestep)

        if mask is not None and masked_latent is not None:
            unconditional_mask = torch.zeros_like(mask)
            unconditional_latent = torch.nn.functional.interpolate(unconditional_mask, size=(masked_latent.shape[2] * 8, masked_latent.shape[3] * 8)).repeat_interleave(3, dim=1)
            unconditional_latent = vae.encode(unconditional_latent)

        time = list(range(self.num_timesteps))[::-1]
        for time in tqdm(time, position=0):
            tensor_time = torch.tensor([time] * shape[0], device=device)
            L = min(self.num_timesteps - time - 1, self.jump_length)
            img = self.inp_q_sample(img, tensor_time, L)
            for i in range(L+1, 0, -1):
                img = self.inp_p_sample(model, img, tensor_time + i - 1, context, context_mask, null_embedding, guidance_weight_text, guidance_weight_image=guidance_weight_image,
                                        mask=mask, masked_latent=masked_latent, image_latent=image_latent, unconditional_mask=unconditional_mask, unconditional_latent=unconditional_latent)           
        return img




def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion
