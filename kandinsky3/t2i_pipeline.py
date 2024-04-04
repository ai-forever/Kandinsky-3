from typing import Union, List
import PIL

import torch
import torchvision.transforms as T
from einops import repeat

from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule


class Kandinsky3T2IPipeline:

    def __init__(
            self,
            device_map: Union[str, torch.device, dict],
            dtype_map: Union[str, torch.dtype, dict],
            unet: UNet,
            null_embedding: torch.Tensor,
            t5_processor: T5TextConditionProcessor,
            t5_encoder: T5TextConditionEncoder,
            movq: MoVQ,
            gan: bool,
    ):
        self.device_map = device_map
        self.dtype_map = dtype_map
        self.to_pil = T.ToPILImage()

        self.unet = unet
        self.null_embedding = null_embedding
        self.t5_processor = t5_processor
        self.t5_encoder = t5_encoder
        self.movq = movq

        self.gan = gan

    def __call__(
            self,
            text: str,
            negative_text: str = None,
            images_num: int = 1,
            bs: int = 1,
            width: int = 1024,
            height: int = 1024,
            guidance_scale: float = 3.0,
            steps: int = 50,
            eta: float = 1.0
    ) -> List[PIL.Image.Image]:

        betas = get_named_beta_schedule('cosine', 1000)
        base_diffusion = BaseDiffusion(betas, 0.99)
        times = list(range(999, 0, -1000 // steps))
        if self.gan:
            times = list(range(979, 0, -250))

        condition_model_input, negative_condition_model_input = self.t5_processor.encode(text, negative_text)
        for input_type in condition_model_input:
            condition_model_input[input_type] = condition_model_input[input_type][None].to(
                self.device_map['text_encoder']
            )

        if negative_condition_model_input is not None:
            for input_type in negative_condition_model_input:
                negative_condition_model_input[input_type] = negative_condition_model_input[input_type][None].to(
                    self.device_map['text_encoder']
                )

        pil_images = []
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype_map['text_encoder']):
                context, context_mask = self.t5_encoder(condition_model_input)
                if negative_condition_model_input is not None:
                    negative_context, negative_context_mask = self.t5_encoder(negative_condition_model_input)
                else:
                    negative_context, negative_context_mask = None, None

            k, m = images_num // bs, images_num % bs
            for minibatch in [bs] * k + [m]:
                if minibatch == 0:
                    continue
                bs_context = repeat(context, '1 n d -> b n d', b=minibatch)
                bs_context_mask = repeat(context_mask, '1 n -> b n', b=minibatch)
                if negative_context is not None:
                    bs_negative_context = repeat(negative_context, '1 n d -> b n d', b=minibatch)
                    bs_negative_context_mask = repeat(negative_context_mask, '1 n -> b n', b=minibatch)
                else:
                    bs_negative_context, bs_negative_context_mask = None, None

                with torch.cuda.amp.autocast(dtype=self.dtype_map['unet']):
                    images = base_diffusion.p_sample_loop(
                        self.unet, (minibatch, 4, height // 8, width // 8), times, self.device_map['unet'],
                        bs_context, bs_context_mask, self.null_embedding, guidance_scale, eta,
                        negative_context=bs_negative_context, negative_context_mask=bs_negative_context_mask,
                        gan=self.gan
                    )

                with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                    images = torch.cat([self.movq.decode(image) for image in images.chunk(2)])
                    images = torch.clip((images + 1.) / 2., 0., 1.)
                    for images_chunk in images.chunk(1):
                        pil_images += [self.to_pil(image) for image in images_chunk]

        return pil_images
