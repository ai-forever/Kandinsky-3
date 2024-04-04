from typing import Union, List
import PIL
import numpy as np

import torch
import torchvision.transforms as T
from einops import repeat

from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule
from kandinsky3.utils import resize_image_for_diffusion, resize_mask_for_diffusion


class Kandinsky3InpaintingPipeline:

    def __init__(
            self,
            device_map: Union[str, torch.device, dict],
            dtype_map: Union[str, torch.dtype, dict],
            unet: UNet,
            null_embedding: torch.Tensor,
            t5_processor: T5TextConditionProcessor,
            t5_encoder: T5TextConditionEncoder,
            movq: MoVQ,
    ):
        self.device_map = device_map
        self.dtype_map = dtype_map
        self.to_pil = T.ToPILImage()
        self.to_tensor = T.ToTensor()

        self.unet = unet
        self.null_embedding = null_embedding
        self.t5_processor = t5_processor
        self.t5_encoder = t5_encoder
        self.movq = movq

    def shared_step(self, batch: dict) -> dict:
        image = batch['image']
        condition_model_input = batch['text']
        negative_condition_model_input = batch['negative_text']

        bs = image.shape[0]

        masked_latent = None
        mask = batch['mask']

        if 'masked_image' in batch:
            masked_latent = batch['masked_image']
        elif self.unet.in_layer.in_channels == 9:
            masked_latent = image.masked_fill((1 - mask).bool(), 0)
        else:
            raise ValueError()

        with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
            masked_latent = self.movq.encode(masked_latent)
        mask = torch.nn.functional.interpolate(mask, size=(masked_latent.shape[2], masked_latent.shape[3]))

        with torch.cuda.amp.autocast(dtype=self.dtype_map['text_encoder']):
            context, context_mask = self.t5_encoder(condition_model_input)

        if negative_condition_model_input is not None:
            negative_context, negative_context_mask = self.t5_encoder(negative_condition_model_input)
        else:
            negative_context, negative_context_mask = None, None

        return {
            'context': context,
            'context_mask': context_mask,
            'negative_context': negative_context,
            'negative_context_mask': negative_context_mask,
            'image': image,
            'masked_latent': masked_latent,
            'mask': mask
        }

    def prepare_batch(
            self,
            text: str,
            negative_text: str,
            image: PIL.Image.Image,
            mask: np.ndarray,
    ) -> dict:
        condition_model_input, negative_condition_model_input = self.t5_processor.encode(
            text=text, negative_text=negative_text
        )
        batch = {
            'image': self.to_tensor(resize_image_for_diffusion(image.convert("RGB"))) * 2 - 1,
            'mask': 1 - self.to_tensor(resize_mask_for_diffusion(mask)),
            'text': condition_model_input,
            'negative_text': negative_condition_model_input
        }
        batch['mask'] = batch['mask'].type(self.dtype_map['movq'])

        batch['image'] = batch['image'].unsqueeze(0).to(self.device_map['movq'])
        batch['text']['input_ids'] = batch['text']['input_ids'].unsqueeze(0).to(self.device_map['text_encoder'])
        batch['text']['attention_mask'] = batch['text']['attention_mask'].unsqueeze(0).to(
            self.device_map['text_encoder'])
        batch['mask'] = batch['mask'].unsqueeze(0).to(self.device_map['movq'])

        if negative_condition_model_input is not None:
            batch['negative_text']['input_ids'] = batch['negative_text']['input_ids'].to(
                self.device_map['text_encoder'])
            batch['negative_text']['attention_mask'] = batch['negative_text']['attention_mask'].to(
                self.device_map['text_encoder'])

        return batch

    def __call__(
            self,
            text: str,
            image: PIL.Image.Image,
            mask: np.ndarray,
            negative_text: str = None,
            images_num: int = 1,
            bs: int = 1,
            steps: int = 50,
            guidance_weight_text: float = 4,
            eta=1.0
    ) -> List[PIL.Image.Image]:

        with torch.no_grad():
            batch = self.prepare_batch(text, negative_text, image, mask)
            processed = self.shared_step(batch)
            betas = get_named_beta_schedule('cosine', 1000)
            base_diffusion = BaseDiffusion(betas, percentile=0.95)
            times = list(range(999, 0, -1000 // steps))

            pil_images = []
            k, m = images_num // bs, images_num % bs
            for minibatch in [bs] * k + [m]:
                if minibatch == 0:
                    continue

                bs_context = repeat(processed['context'], '1 n d -> b n d', b=minibatch)
                bs_context_mask = repeat(processed['context_mask'], '1 n -> b n', b=minibatch)

                if processed['negative_context'] is not None:
                    bs_negative_context = repeat(processed['negative_context'], '1 n d -> b n d', b=minibatch)
                    bs_negative_context_mask = repeat(processed['negative_context_mask'], '1 n -> b n', b=minibatch)
                else:
                    bs_negative_context, bs_negative_context_mask = None, None

                mask = processed['mask'].repeat_interleave(minibatch, dim=0)
                masked_latent = processed['masked_latent'].repeat_interleave(minibatch, dim=0)

                minibatch = masked_latent.shape[0]

                with torch.cuda.amp.autocast(dtype=self.dtype_map['unet']):
                    with torch.no_grad():
                        images = base_diffusion.p_sample_loop(
                            self.unet, (minibatch, 4, masked_latent.shape[2], masked_latent.shape[3]), times,
                            self.device_map['unet'],
                            bs_context, bs_context_mask, self.null_embedding, guidance_weight_text, eta,
                            negative_context=bs_negative_context, negative_context_mask=bs_negative_context_mask,
                            mask=mask, masked_latent=masked_latent, gan=False
                        )

                with torch.cuda.amp.autocast(dtype=self.dtype_map['movq']):
                    images = torch.cat([self.movq.decode(image) for image in images.chunk(2)])
                    images = torch.clip((images + 1.) / 2., 0., 1.).cpu()

                for images_chunk in images.chunk(1):
                    pil_images += [self.to_pil(image) for image in images_chunk]

        return pil_images
