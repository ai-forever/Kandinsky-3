import math

import torch
from einops import rearrange
from tqdm import tqdm

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

    def get_x_start(self, x, t, noise):
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, noise.shape)
        pred_x_start = (x - sqrt_one_minus_alphas_cumprod * noise) / sqrt_alphas_cumprod
        return pred_x_start

    def get_noise(self, x, t, x_start):
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        pred_noise = (x - sqrt_alphas_cumprod * x_start) / sqrt_one_minus_alphas_cumprod
        return pred_noise

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = self.gen_noise(x_start)
        sqrt_alphas_cumprod = get_tensor_items(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod = get_tensor_items(self.sqrt_one_minus_alphas_cumprod, t, noise.shape)
        x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise
        return x_t

    def q_posterior_mean_variance(self, x_start, x_t, t):
        posterior_mean_coef_1 = get_tensor_items(self.posterior_mean_coef_1, t, x_start.shape)
        posterior_mean_coef_2 = get_tensor_items(self.posterior_mean_coef_2, t, x_t.shape)
        posterior_mean = posterior_mean_coef_1 * x_start + posterior_mean_coef_2 * x_t

        posterior_variance = get_tensor_items(self.posterior_variance, t, x_start.shape)
        posterior_log_variance = get_tensor_items(self.posterior_log_variance, t, x_start.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def q_posterior_variance(self, t, prev_t, shape, eta=1., ):
        alphas_cumprod = get_tensor_items(self.alphas_cumprod, t, shape)
        prev_alphas_cumprod = get_tensor_items(self.alphas_cumprod, prev_t, shape)

        posterior_variance = torch.sqrt(
            eta * (1. - alphas_cumprod / prev_alphas_cumprod) * (1. - prev_alphas_cumprod) / (1. - alphas_cumprod)
        )
        return posterior_variance

    def text_guidance(
            self, model, x, t, context, context_mask, null_embedding, guidance_weight_text,
            uncondition_context=None, uncondition_context_mask=None, mask=None, masked_latent=None
    ):
        large_x = x.repeat(2, 1, 1, 1)
        large_t = t.repeat(2).to(x.dtype)

        if uncondition_context is None:
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

        if model.in_layer.in_channels == 9:
            large_x = torch.cat([large_x, mask, masked_latent], dim=1)

        pred_large_noise = model(large_x, large_t * self.time_scale, large_context, large_context_mask.bool())
        pred_noise, uncond_pred_noise = torch.chunk(pred_large_noise, 2)
        pred_noise = (guidance_weight_text + 1.) * pred_noise - guidance_weight_text * uncond_pred_noise
        return pred_noise

    def p_mean_variance(
            self, model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
            negative_context=None, negative_context_mask=None, mask=None, masked_latent=None
    ):

        pred_noise = self.text_guidance(
            model, x, t, context, context_mask, null_embedding, guidance_weight_text,
            negative_context, negative_context_mask, mask, masked_latent
        )

        pred_x_start = self.get_x_start(x, t, pred_noise)
        pred_x_start = self.process_x_start(pred_x_start)
        pred_noise = self.get_noise(x, t, pred_x_start)
        pred_var = self.q_posterior_variance(t, prev_t, x.shape, eta)

        prev_alphas_cumprod = get_tensor_items(self.alphas_cumprod, prev_t, x.shape)
        pred_mean = torch.sqrt(prev_alphas_cumprod) * pred_x_start
        pred_mean += torch.sqrt(1. - prev_alphas_cumprod - pred_var ** 2) * pred_noise
        return pred_mean, pred_var

    @torch.no_grad()
    def p_sample(
            self, model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
            negative_context=None, negative_context_mask=None, mask=None, masked_latent=None
    ):
        bs = x.shape[0]
        ndims = len(x.shape[1:])
        pred_mean, pred_var = self.p_mean_variance(
            model, x, t, prev_t, context, context_mask, null_embedding, guidance_weight_text, eta,
            negative_context=negative_context, negative_context_mask=negative_context_mask,
            mask=mask, masked_latent=masked_latent
        )
        noise = torch.randn_like(x)
        mask = (prev_t != 0).reshape(bs, *((1,) * ndims))
        sample = pred_mean + mask * pred_var * noise
        return sample

    @torch.no_grad()
    def p_sample_loop(
            self, model, shape, times, device, context, context_mask, null_embedding, guidance_weight_text, eta=1.,
            negative_context=None, negative_context_mask=None, mask=None, masked_latent=None, gan=False,
    ):
        img = torch.randn(*shape, device=device)
        times = times + [0, ]
        times = list(zip(times[:-1], times[1:]))

        for time, prev_time in tqdm(times):
            time = torch.tensor([time] * shape[0], device=device)
            if gan:
                x_t = self.q_sample(img, time)
                pred_noise = model(x_t, time.type(x_t.dtype), context, context_mask.bool())
                img = self.get_x_start(x_t, time, pred_noise)
            else:
                prev_time = torch.tensor([prev_time] * shape[0], device=device)
                img = self.p_sample(
                    model, img, time, prev_time, context, context_mask, null_embedding, guidance_weight_text, eta,
                    negative_context=negative_context, negative_context_mask=negative_context_mask,
                    mask=mask, masked_latent=masked_latent
                )
        return img


def get_diffusion(conf):
    betas = get_named_beta_schedule(**conf.schedule_params)
    base_diffusion = BaseDiffusion(betas, **conf.diffusion_params)
    return base_diffusion
