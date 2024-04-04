import os
from typing import Optional, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download

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
        dtype: Union[str, torch.dtype] = torch.float32,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
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

    null_embedding = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        null_embedding = state_dict['null_embedding']
        unet.load_state_dict(state_dict['unet'])

    unet.to(device=device, dtype=dtype).eval()
    return unet, null_embedding


def get_T5encoder(
        device: Union[str, torch.device],
        weights_path: str,
        projection_name: str,
        dtype: Union[str, torch.dtype] = torch.float32,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
) -> (T5TextConditionProcessor, T5TextConditionEncoder):
    tokens_length = 128
    context_dim = 4096
    processor = T5TextConditionProcessor(tokens_length, weights_path)
    condition_encoder = T5TextConditionEncoder(
        weights_path, context_dim, low_cpu_mem_usage=low_cpu_mem_usage, device=device,
        dtype=dtype, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )

    if weights_path:
        projections_weights_path = os.path.join(weights_path, projection_name)
        state_dict = torch.load(projections_weights_path, map_location=torch.device('cpu'))
        condition_encoder.projection.load_state_dict(state_dict)

    condition_encoder.projection.to(device=device, dtype=dtype).eval()
    return processor, condition_encoder


def get_movq(
        device: Union[str, torch.device],
        weights_path: Optional[str] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
) -> MoVQ:
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

    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        movq.load_state_dict(state_dict)

    movq.to(device=device, dtype=dtype).eval()
    return movq


def get_inpainting_unet(
        device: Union[str, torch.device],
        weights_path: Optional[str] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
) -> (UNet, Optional[torch.Tensor], Optional[dict]):
    unet = UNet(
        model_channels=384,
        num_channels=9,
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

    null_embedding = None
    if weights_path:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        null_embedding = state_dict['null_embedding']
        unet.load_state_dict(state_dict['unet'])

    unet.to(device=device, dtype=dtype).eval()
    return unet, null_embedding


def get_T2I_pipeline(
        device_map: Union[str, torch.device, dict],
        dtype_map: Union[str, torch.dtype, dict] = torch.float32,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        cache_dir: str = '/tmp/kandinsky3/',
        unet_path: str = None,
        text_encoder_path: str = None,
        movq_path: str = None,
) -> Kandinsky3T2IPipeline:
    # assert ((unet_path is not None) or (text_encoder_path is not None) or (movq_path is not None))
    if not isinstance(device_map, dict):
        device_map = {
            'unet': device_map, 'text_encoder': device_map, 'movq': device_map
        }
    if not isinstance(dtype_map, dict):
        dtype_map = {
            'unet': dtype_map, 'text_encoder': dtype_map, 'movq': dtype_map
        }

    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/kandinsky3.pt', cache_dir=cache_dir
        )
    if text_encoder_path is None:
        text_encoder_path = snapshot_download(
            repo_id="ai-forever/Kandinsky3.1", allow_patterns="weights/flan_ul2_encoder/*", cache_dir=cache_dir
        )
        text_encoder_path = os.path.join(text_encoder_path, 'weights/flan_ul2_encoder')
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/movq.pt', cache_dir=cache_dir
        )

    unet, null_embedding = get_T2I_unet(device_map['unet'], unet_path, dtype=dtype_map['unet'])
    processor, condition_encoder = get_T5encoder(
        device_map['text_encoder'], text_encoder_path, 'projection.pt', dtype=dtype_map['text_encoder'],
        low_cpu_mem_usage=low_cpu_mem_usage, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )
    movq = get_movq(device_map['movq'], movq_path, dtype=dtype_map['movq'])
    return Kandinsky3T2IPipeline(
        device_map, dtype_map, unet, null_embedding, processor, condition_encoder, movq, False
    )


def get_T2I_Flash_pipeline(
        device_map: Union[str, torch.device, dict],
        dtype_map: Union[str, torch.dtype, dict] = torch.float32,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        cache_dir: str = '/tmp/kandinsky3/',
        unet_path: str = None,
        text_encoder_path: str = None,
        movq_path: str = None,
) -> Kandinsky3T2IPipeline:
    # assert ((unet_path is not None) or (text_encoder_path is not None) or (movq_path is not None))
    if not isinstance(device_map, dict):
        device_map = {
            'unet': device_map, 'text_encoder': device_map, 'movq': device_map
        }
    if not isinstance(dtype_map, dict):
        dtype_map = {
            'unet': dtype_map, 'text_encoder': dtype_map, 'movq': dtype_map
        }

    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/kandinsky3_flash.pt', cache_dir=cache_dir
        )
    if text_encoder_path is None:
        text_encoder_path = snapshot_download(
            repo_id="ai-forever/Kandinsky3.1", allow_patterns="weights/flan_ul2_encoder/*", cache_dir=cache_dir
        )
        text_encoder_path = os.path.join(text_encoder_path, 'weights/flan_ul2_encoder')
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/movq.pt', cache_dir=cache_dir
        )

    unet, null_embedding = get_T2I_unet(device_map['unet'], unet_path, dtype=dtype_map['unet'])
    processor, condition_encoder = get_T5encoder(
        device_map['text_encoder'], text_encoder_path, 'projection_flash.pt', dtype=dtype_map['text_encoder'],
        low_cpu_mem_usage=low_cpu_mem_usage, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )
    movq = get_movq(device_map['movq'], movq_path, dtype=dtype_map['movq'])
    return Kandinsky3T2IPipeline(
        device_map, dtype_map, unet, null_embedding, processor, condition_encoder, movq, True
    )


def get_inpainting_pipeline(
        device_map: Union[str, torch.device, dict],
        dtype_map: Union[str, torch.dtype, dict] = torch.float32,
        low_cpu_mem_usage: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        cache_dir: str = '/tmp/kandinsky3/',
        unet_path: str = None,
        text_encoder_path: str = None,
        movq_path: str = None,
) -> Kandinsky3InpaintingPipeline:
    # assert ((unet_path is not None) or (text_encoder_path is not None) or (movq_path is not None))
    if not isinstance(device_map, dict):
        device_map = {
            'unet': device_map, 'text_encoder': device_map, 'movq': device_map
        }
    if not isinstance(dtype_map, dict):
        dtype_map = {
            'unet': dtype_map, 'text_encoder': dtype_map, 'movq': dtype_map
        }

    if unet_path is None:
        unet_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/kandinsky3_inpainting.pt', cache_dir=cache_dir
        )
    if text_encoder_path is None:
        text_encoder_path = snapshot_download(
            repo_id="ai-forever/Kandinsky3.1", allow_patterns="weights/flan_ul2_encoder/*", cache_dir=cache_dir
        )
        text_encoder_path = os.path.join(text_encoder_path, 'weights/flan_ul2_encoder')
    if movq_path is None:
        movq_path = hf_hub_download(
            repo_id="ai-forever/Kandinsky3.1", filename='weights/movq.pt', cache_dir=cache_dir
        )

    unet, null_embedding = get_inpainting_unet(device_map['unet'], unet_path, dtype=dtype_map['unet'])
    processor, condition_encoder = get_T5encoder(
        device_map['text_encoder'], text_encoder_path, 'projection_inpainting.pt', dtype=dtype_map['text_encoder'],
        low_cpu_mem_usage=low_cpu_mem_usage, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit
    )
    movq = get_movq(device_map['movq'], movq_path, dtype=dtype_map['movq'])
    return Kandinsky3InpaintingPipeline(
        device_map, dtype_map, unet, null_embedding, processor, condition_encoder, movq
    )
