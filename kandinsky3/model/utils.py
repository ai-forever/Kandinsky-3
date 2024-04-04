from torch.nn import Identity
from einops import rearrange


def exist(item):
    return item is not None


def set_default_item(condition, item_1, item_2=None):
    if condition:
        return item_1
    else:
        return item_2


def set_default_layer(condition, layer_1, args_1=[], kwargs_1={}, layer_2=Identity, args_2=[], kwargs_2={}):
    if condition:
        return layer_1(*args_1, **kwargs_1)
    else:
        return layer_2(*args_2, **kwargs_2)


def get_tensor_items(x, pos, broadcast_shape):
    device = pos.device
    bs = pos.shape[0]
    ndims = len(broadcast_shape[1:])
    x = x.cpu()[pos.cpu()]
    return x.reshape(bs, *((1,) * ndims)).to(device)


def local_patching(x, height, width, group_size):
    if group_size > 0:
        x = rearrange(
            x, 'b c (h g1) (w g2) -> b (h w) (g1 g2) c',
            h=height//group_size, w=width//group_size, g1=group_size, g2=group_size
        )
    else:
        x = rearrange(x, 'b c h w -> b (h w) c', h=height, w=width)
    return x


def local_merge(x, height, width, group_size):
    if group_size > 0:
        x = rearrange(
            x, 'b (h w) (g1 g2) c -> b c (h g1) (w g2)',
            h=height//group_size, w=width//group_size, g1=group_size, g2=group_size
        )
    else:
        x = rearrange(x, 'b (h w) c -> b c h w', h=height, w=width)
    return x


def global_patching(x, height, width, group_size):
    x = local_patching(x, height, width, height//group_size)
    x = x.transpose(-2, -3)
    return x


def global_merge(x, height, width, group_size):
    x = x.transpose(-2, -3)
    x = local_merge(x, height, width, height//group_size)
    return x
