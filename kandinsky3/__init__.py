from typing import Optional, Union, List
import PIL
import os

import numpy as np
from tqdm import tqdm

import torch
import omegaconf
from omegaconf import OmegaConf
from kandinsky3.model.unet import UNet
from kandinsky3.movq import MoVQ
from kandinsky3.condition_encoders import T5TextConditionEncoder
from kandinsky3.condition_processors import T5TextConditionProcessor
from kandinsky3.model.diffusion import BaseDiffusion, get_named_beta_schedule

from .t2i_pipeline import Kandinsky3T2IPipeline
from .inpainting_pipeline import Kandinsky3InpaintingPipeline


def get_T2I_unet(
    device: Union[str, torch.device],
    weights_path: Optional[str] = None, 
    fp16: bool = False
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    unet = UNet(
        model_channels=384,
        num_channels=4,
        init_channels=192,
        time_embed_dim=1536,
        context_dim=4096,
        groups=32,
        head_dim=64,
        expansion_ratio=4,
        compression_ratio=2,
        dim_mult=(1, 2, 4, 8),
        num_blocks=(3, 3, 3, 3),
        add_cross_attention=(False, True, True, True),
        add_self_attention=(False, True, True, True),
    )

    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        ema_state_dict = {}
        for file_name in tqdm(os.listdir(weights_path)):
            ema_state_dict.update(torch.load(
                os.path.join(weights_path, file_name), 
                map_location=torch.device('cpu')
            ))

        projections_state_dict = {
            key.replace('encoder.projections.', ''): value
            for key, value in ema_state_dict.items()  if 'projections' in key
        }
        null_embedding = {
            key.replace('null_embedding.', ''): value
            for key, value in ema_state_dict.items()  if 'null_embedding' in key
        }['null_embedding']
        unet.load_state_dict({
            key.replace('unet.', ''): value
            for key, value in ema_state_dict.items()  if 'unet' in key
        })
    
    unet.eval().to(device)

    if fp16:
        unet = unet.half()

    return unet, null_embedding, projections_state_dict


def get_T5encoder(
    device: Union[str, torch.device],
    weights_path: str, 
    projections_state_dict: Optional[dict] = None,
    fp16: bool = True,
    low_cpu_mem_usage: bool = True,
    device_map: Optional[str] = None
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    model_names = {'t5': weights_path}
    tokens_length = {'t5': 128}
    context_dim = 4096
    model_dims = {'t5': 4096}
    processor = T5TextConditionProcessor(tokens_length, model_names)
    condition_encoders = T5TextConditionEncoder(
        model_names, context_dim, model_dims, low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map
    )
    
    if projections_state_dict:
        condition_encoders.projections.load_state_dict(projections_state_dict)
        
    condition_encoders = condition_encoders.eval().to(device)
    return processor, condition_encoders


def get_movq(
    device: Union[str, torch.device],
    weights_path: str, 
    fp16: bool = False
) -> (UNet, Optional[dict], Optional[torch.Tensor]):
    generator_config = {
        'double_z': False,
        'z_channels': 4,
        'resolution': 256,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 256,
        'ch_mult': [1, 2, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [32],
        'dropout': 0.0
    }
    movq = MoVQ(generator_config)
    movq.load_state_dict(torch.load(weights_path))
    movq = movq.eval().to(device)

    if fp16:
        movq = movq.half()
    return movq


def get_inpainting_unet(
    device: Union[str, torch.device],
    config_path: str,
    weights_path: Optional[str] = None, 
    fp16: bool = False
) -> (omegaconf.DictConfig, UNet, Optional[dict], Optional[torch.Tensor]):
    conf = OmegaConf.load(config_path)
    inpaint_unet = UNet(**conf.model.unet_params)

    # load weights
    null_embedding = None
    projections_state_dict = None
    if weights_path:
        ema_state_dict = {}

        if os.path.isdir(weights_path):
            for file_name in os.listdir(weights_path):
                ema_state_dict.update(torch.load(os.path.join(weights_path, file_name), map_location='cpu'))
        elif os.path.isfile(weights_path):
            ema_state_dict = torch.load(weights_path, map_location='cpu')
            if 'state_dict' in ema_state_dict:
                ema_state_dict = ema_state_dict['state_dict']

        projections_state_dict = {
            key.replace('encoder.projections.', ''): value
            for key, value in ema_state_dict.items()  if 'projections' in key
        }
        null_embedding = {
            key.replace('null_embedding.', ''): value
            for key, value in ema_state_dict.items()  if 'null_embedding' in key
        }['null_embedding']
        unet_state_dict = {
            key.replace('unet.', ''): value
            for key, value in ema_state_dict.items()  if 'unet' in key
        }
        
        if unet_state_dict['in_layer.weight'].shape[1] == 4:
            unet.state_dict()['in_layer.weight'].zero_()
            unet.state_dict()['in_layer.weight'][:, :4] = unet_state_dict['in_layer.weight']
            del unet_state_dict['in_layer.weight']
        
        inpaint_unet.load_state_dict(unet_state_dict, strict=False)
    
    if fp16:
        inpaint_unet = inpaint_unet.half()
    
    inpaint_unet.eval().to(device)

    return conf, inpaint_unet, null_embedding, projections_state_dict
