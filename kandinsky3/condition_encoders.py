from abc import abstractmethod
from typing import Optional
import torch
from torch import nn
from einops import repeat
from transformers import T5Model, CLIPModel
from typing import Optional

from .utils import freeze


class ConditionEncoder(nn.Module):

    def __init__(self, context_dim, model_dims):
        super().__init__()
        self.model_idx = {key: i for i, key in enumerate(model_dims.keys())}
        self.projections = nn.ModuleDict({
            model_name: nn.Sequential(
                nn.Linear(model_dim, context_dim, bias=False),
                nn.LayerNorm(context_dim)
            ) for model_name, model_dim in model_dims.items()
        })

    @abstractmethod
    def encode(self, model_input, model_name):
        pass

    def forward(self, model_inputs):
        context = []
        context_mask = []
        for model_name, model_idx in self.model_idx.items():
            model_input = model_inputs[model_name]
            embeddings = self.encode(model_input, model_name)
            if 'attention_mask' in model_input:
                bad_embeddings = (embeddings == 0).all(-1).all(-1)
                model_input['attention_mask'][bad_embeddings] = torch.zeros_like(model_input['attention_mask'][bad_embeddings])
            embeddings = self.projections[model_name](embeddings)
            if 'attention_mask' in model_input:
                attention_mask = model_input['attention_mask']
                embeddings[attention_mask == 0] = torch.zeros_like(embeddings[attention_mask == 0])
                max_seq_length = attention_mask.sum(-1).max() + 1
                embeddings = embeddings[:, :max_seq_length]
                attention_mask = attention_mask[:, :max_seq_length]
            else:
                attention_mask = torch.ones(*embeddings.shape[:-1], dtype=torch.long, device=embeddings.device)
            context.append(embeddings)
            context_mask.append((model_idx + 1) * attention_mask)
        context = torch.cat(context, dim=1)
        context_mask = torch.cat(context_mask, dim=1)
        return context, context_mask


class T5TextConditionEncoder(ConditionEncoder):

    def __init__(self, model_names, context_dim, model_dims, low_cpu_mem_usage: bool = True, device_map: Optional[str] = None):
        super().__init__(context_dim, model_dims)
        t5_model = T5Model.from_pretrained(model_names['t5'], low_cpu_mem_usage=low_cpu_mem_usage, device_map=device_map)
        self.encoders = nn.ModuleDict({
            't5': t5_model.encoder.half(),
        })
        self.encoders = freeze(self.encoders)

    @torch.no_grad()
    def encode(self, model_input, model_name):
        embeddings = self.encoders[model_name](**model_input).last_hidden_state
        is_inf_embeddings = torch.isinf(embeddings).any(-1).any(-1)
        is_nan_embeddings = torch.isnan(embeddings).any(-1).any(-1)
        bad_embeddings = is_inf_embeddings + is_nan_embeddings
        embeddings[bad_embeddings] = torch.zeros_like(embeddings[bad_embeddings])
        embeddings = embeddings.type(torch.float32)
        return embeddings


def get_condition_encoder(conf):
    if hasattr(conf, 'model_names'):
        model_names = conf.model_names.keys()
        if 't5' in model_names:
            return T5TextConditionEncoder(**conf)
        else:
            raise NotImplementedError("Condition Encoder not implemented")
    else:
        return ConditionEncoder(**conf)
