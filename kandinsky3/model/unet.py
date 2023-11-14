import torch
from torch import nn, einsum
from einops import rearrange

from .nn import Identity, Attention, SinusoidalPosEmb, ConditionalGroupNorm
from .utils import exist, set_default_item, set_default_layer
import torch.nn.functional as F


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, time_embed_dim, kernel_size=3, norm_groups=32, up_resolution=None):
        super().__init__()
        self.group_norm = ConditionalGroupNorm(norm_groups, in_channels, time_embed_dim)
        self.activation = nn.SiLU()
        self.up_sample = set_default_layer(
            exist(up_resolution) and up_resolution,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
        padding = set_default_item(kernel_size == 1, 0, 1)
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.down_sample = set_default_layer(
            exist(up_resolution) and not up_resolution,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )

    def forward(self, x, time_embed):
        x = self.group_norm(x, time_embed)
        x = self.activation(x)
        x = self.up_sample(x)
        x = self.projection(x)
        x = self.down_sample(x)
        return x


class ResNetBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, norm_groups=32, compression_ratio=2, up_resolutions=4*[None]
    ):
        super().__init__()
        kernel_sizes = [1, 3, 3, 1]
        hidden_channel = max(in_channels, out_channels) // compression_ratio
        hidden_channels = [(in_channels, hidden_channel)] + [(hidden_channel, hidden_channel)] * 2 + [(hidden_channel, out_channels)]
        self.resnet_blocks = nn.ModuleList([
            Block(in_channel, out_channel, time_embed_dim, kernel_size, norm_groups, up_resolution)
            for (in_channel, out_channel), kernel_size, up_resolution in zip(hidden_channels, kernel_sizes, up_resolutions)
        ])

        self.shortcut_up_sample = set_default_layer(
            True in up_resolutions,
            nn.ConvTranspose2d, (in_channels, in_channels), {'kernel_size': 2, 'stride': 2}
        )
        self.shortcut_projection = set_default_layer(
            in_channels != out_channels,
            nn.Conv2d, (in_channels, out_channels), {'kernel_size': 1}
        )
        self.shortcut_down_sample = set_default_layer(
            False in up_resolutions,
            nn.Conv2d, (out_channels, out_channels), {'kernel_size': 2, 'stride': 2}
        )

    def forward(self, x, time_embed):
        out = x
        for resnet_block in self.resnet_blocks:
            out = resnet_block(out, time_embed)

        x = self.shortcut_up_sample(x)
        x = self.shortcut_projection(x)
        x = self.shortcut_down_sample(x)
        x = x + out
        return x


class AttentionPolling(nn.Module):

    def __init__(self, num_channels, context_dim, head_dim=64):
        super().__init__()
        self.attention = Attention(context_dim, num_channels, context_dim, head_dim)

    def forward(self, x, context, context_mask=None):
        context = self.attention(context.mean(dim=1, keepdim=True), context, context_mask)
        return x + context.squeeze(1)


class AttentionBlock(nn.Module):

    def __init__(self, num_channels, time_embed_dim, context_dim=None, norm_groups=32, head_dim=64, expansion_ratio=4):
        super().__init__()
        self.in_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.attention = Attention(num_channels, num_channels, context_dim or num_channels, head_dim)

        hidden_channels = expansion_ratio * num_channels
        self.out_norm = ConditionalGroupNorm(norm_groups, num_channels, time_embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Conv2d(num_channels, hidden_channels, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, num_channels, kernel_size=1, bias=False),
        )

    def forward(self, x, time_embed, context=None, context_mask=None):
        height, width = x.shape[-2:]
        out = self.in_norm(x, time_embed)
        out = rearrange(out, 'b c h w -> b (h w) c', h=height, w=width)
        context = set_default_item(exist(context), context, out)
        out = self.attention(out, context, context_mask)
        out = rearrange(out, 'b (h w) c -> b c h w', h=height, w=width)
        x = x + out

        out = self.out_norm(x, time_embed)
        out = self.feed_forward(out)
        x = x + out
        return x


class DownSampleBlock(nn.Module):

    def __init__(
            self, in_channels, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            down_sample=True, self_attention=True
    ):
        super().__init__()
        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (in_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

        up_resolutions = [[None] * 4] * (num_blocks - 1) + [[None, None, set_default_item(down_sample, False), None]]
        hidden_channels = [(in_channels, out_channels)] + [(out_channels, out_channels)] * (num_blocks - 1)
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (out_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=Identity
                ),
                ResNetBlock(out_channel, out_channel, time_embed_dim, groups, compression_ratio, up_resolution),
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

    def forward(self, x, time_embed, context=None, context_mask=None, control_net_residual=None):
        x = self.self_attention_block(x, time_embed)
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed)
            x = attention(x, time_embed, context, context_mask)
            x = out_resnet_block(x, time_embed)
        return x


class UpSampleBlock(nn.Module):

    def __init__(
            self, in_channels, cat_dim, out_channels, time_embed_dim, context_dim=None,
            num_blocks=3, groups=32, head_dim=64, expansion_ratio=4, compression_ratio=2,
            up_sample=True, self_attention=True
    ):
        super().__init__()
        up_resolutions = [[None, set_default_item(up_sample, True), None, None]] + [[None] * 4] * (num_blocks - 1)
        hidden_channels = [(in_channels + cat_dim, in_channels)] + [(in_channels, in_channels)] * (num_blocks - 2) + [(in_channels, out_channels)]
        self.resnet_attn_blocks = nn.ModuleList([
            nn.ModuleList([
                ResNetBlock(in_channel, in_channel, time_embed_dim, groups, compression_ratio, up_resolution),
                set_default_layer(
                    exist(context_dim),
                    AttentionBlock,
                    (in_channel, time_embed_dim, context_dim, groups, head_dim, expansion_ratio),
                    layer_2=Identity
                ),
                ResNetBlock(in_channel, out_channel, time_embed_dim, groups, compression_ratio),
            ]) for (in_channel, out_channel), up_resolution in zip(hidden_channels, up_resolutions)
        ])

        self.self_attention_block = set_default_layer(
            self_attention,
            AttentionBlock,
            (out_channels, time_embed_dim, None, groups, head_dim, expansion_ratio),
            layer_2=Identity
        )

    def forward(self, x, time_embed, context=None, context_mask=None):
        for in_resnet_block, attention, out_resnet_block in self.resnet_attn_blocks:
            x = in_resnet_block(x, time_embed)
            x = attention(x, time_embed, context, context_mask)
            x = out_resnet_block(x, time_embed)
        x = self.self_attention_block(x, time_embed)
        return x

class ControlNetModel(nn.Module):
    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 out_channels=4,
                 time_embed_dim=None,
                 context_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True)
                 ):
        super().__init__()
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

    def forward(self, x, time, context=None, context_mask=None):
        time_embed = self.to_time_embed(time)
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        return hidden_states

class UNet(nn.Module):

    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 out_channels=4,
                 time_embed_dim=None,
                 context_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim,
                    expansion_ratio, compression_ratio, up_sample, self_attention
                )
            )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)
        )

        self.control_net = None

    def forward(self, x, time, context=None, context_mask=None, control_net_residual=None):
        time_embed = self.to_time_embed(time)
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask, control_net_residual)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask)
        x = self.out_layer(x)
        return x


class ControlNetModel(nn.Module):
    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 out_channels=4,
                 time_embed_dim=None,
                 context_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                *args,
                 **kwargs,
                 ):
        super().__init__()
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

    def forward(self, x, time, context=None, context_mask=None):
        time_embed = self.to_time_embed(time)
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        hidden_states = []
        x = self.in_layer(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask)
            if level != self.num_levels - 1:
                hidden_states.append(x)
        return hidden_states

class ControlUNet(nn.Module):

    def __init__(self,
                 model_channels,
                 init_channels=None,
                 num_channels=3,
                 out_channels=4,
                 time_embed_dim=None,
                 context_dim=None,
                 groups=32,
                 head_dim=64,
                 expansion_ratio=4,
                 compression_ratio=2,
                 dim_mult=(1, 2, 4, 8),
                 num_blocks=(3, 3, 3, 3),
                 add_cross_attention=(False, True, True, True),
                 add_self_attention=(False, True, True, True),
                 control_net_channels=5,
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        init_channels = init_channels or model_channels
        self.to_time_embed = nn.Sequential(
            SinusoidalPosEmb(init_channels),
            nn.Linear(init_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.feature_pooling = AttentionPolling(time_embed_dim, context_dim, head_dim)

        self.in_layer = nn.Conv2d(num_channels, init_channels, kernel_size=3, padding=1)

        hidden_dims = [init_channels, *map(lambda mult: model_channels * mult, dim_mult)]
        in_out_dims = list(zip(hidden_dims[:-1], hidden_dims[1:]))
        text_dims = [set_default_item(is_exist, context_dim) for is_exist in add_cross_attention]
        layer_params = [num_blocks, text_dims, add_self_attention]
        rev_layer_params = map(reversed, layer_params)

        cat_dims = []
        self.num_levels = len(in_out_dims)
        self.down_samples = nn.ModuleList([])
        for level, ((in_dim, out_dim), res_block_num, text_dim, self_attention) in enumerate(zip(in_out_dims, *layer_params)):
            down_sample = level != (self.num_levels - 1)
            cat_dims.append(set_default_item(level != (self.num_levels - 1), out_dim, 0))
            self.down_samples.append(
                DownSampleBlock(
                    in_dim, out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim, expansion_ratio,
                    compression_ratio, down_sample, self_attention
                )
            )

        self.up_samples = nn.ModuleList([])
        for level, ((out_dim, in_dim), res_block_num, text_dim, self_attention) in enumerate(zip(reversed(in_out_dims), *rev_layer_params)):
            up_sample = level != 0
            self.up_samples.append(
                UpSampleBlock(
                    in_dim, cat_dims.pop(), out_dim, time_embed_dim, text_dim, res_block_num, groups, head_dim,
                    expansion_ratio, compression_ratio, up_sample, self_attention
                )
            )

        self.out_layer = nn.Sequential(
            nn.GroupNorm(groups, init_channels),
            nn.SiLU(),
            nn.Conv2d(init_channels, out_channels, kernel_size=3, padding=1)
        )

        self.control_net = ControlNetModel(model_channels,
                                            init_channels,
                                            control_net_channels,
                                            out_channels,
                                            time_embed_dim,
                                            context_dim,
                                            groups,
                                            head_dim,
                                            expansion_ratio,
                                            compression_ratio,
                                            dim_mult,
                                            num_blocks,
                                            add_cross_attention,
                                            add_self_attention)

    def forward(self, x, time, context=None, context_mask=None, control_net_data=None):
        time_embed = self.to_time_embed(time)
        if exist(context):
            time_embed = self.feature_pooling(time_embed, context, context_mask)

        control_net_hiddens =  self.control_net(control_net_data, time, context, context_mask)
        hidden_states = []
        x = self.in_layer(x)
        for level, down_sample in enumerate(self.down_samples):
            x = down_sample(x, time_embed, context, context_mask)
            if level != self.num_levels - 1:
                x += control_net_hiddens.pop(0)
                hidden_states.append(x)
        for level, up_sample in enumerate(self.up_samples):
            if level != 0:
                x = torch.cat([x, hidden_states.pop()], dim=1)
            x = up_sample(x, time_embed, context, context_mask)
        x = self.out_layer(x)
        return x


def get_control_unet(conf):
    unet = ControlUNet(**conf)
    return unet


def get_unet(conf):
    unet = UNet(**conf)
    return unet
