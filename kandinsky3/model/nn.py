import math

import torch
from torch import nn, einsum
from einops import rearrange, repeat

from .utils import exist


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @staticmethod
    def forward(x, *args, **kwargs):
        return x


class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class ConditionalGroupNorm(nn.Module):

    def __init__(self, groups, normalized_shape, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(groups, normalized_shape, affine=False)
        self.context_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_dim, 2 * normalized_shape)
        )
        self.context_mlp[1].weight.data.zero_()
        self.context_mlp[1].bias.data.zero_()

    def forward(self, x, context):
        context = self.context_mlp(context)
        ndims = ' 1' * len(x.shape[2:])
        context = rearrange(context, f'b c -> b c{ndims}')

        scale, shift = context.chunk(2, dim=1)
        x = self.norm(x) * (scale + 1.) + shift
        return x


class Attention(nn.Module):

    def __init__(self, in_channels, out_channels, context_dim, head_dim=64):
        super().__init__()
        assert out_channels % head_dim == 0
        self.num_heads = out_channels // head_dim
        self.scale = head_dim ** -0.5

        self.to_query = nn.Linear(in_channels, out_channels, bias=False)
        self.to_key = nn.Linear(context_dim, out_channels, bias=False)
        self.to_value = nn.Linear(context_dim, out_channels, bias=False)

        self.output_layer = nn.Linear(out_channels, out_channels, bias=False)

    def forward(self, x, context, context_mask=None):
        query = rearrange(self.to_query(x), 'b n (h d) -> b h n d', h=self.num_heads)
        key = rearrange(self.to_key(context), 'b n (h d) -> b h n d', h=self.num_heads)
        value = rearrange(self.to_value(context), 'b n (h d) -> b h n d', h=self.num_heads)

        attention_matrix = einsum('b h i d, b h j d -> b h i j', query, key) * self.scale
        if exist(context_mask):
            max_neg_value = -torch.finfo(attention_matrix.dtype).max
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            attention_matrix = attention_matrix.masked_fill(~context_mask, max_neg_value)
        attention_matrix = attention_matrix.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attention_matrix, value)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.output_layer(out)
        return out
