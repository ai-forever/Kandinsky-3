from omegaconf import OmegaConf
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.path import Path
import torch.nn as nn
from skimage.transform import resize


def load_conf(config_path):
    conf = OmegaConf.load(config_path)
    conf.data.tokens_length = conf.common.tokens_length
    conf.data.processor_names = conf.model.encoders.model_names
    conf.data.dataset.seed = conf.common.seed
    conf.data.dataset.image_size = conf.common.image_size

    conf.trainer.trainer_params.max_steps = conf.common.train_steps
    conf.scheduler.params.total_steps = conf.common.train_steps
    conf.logger.tensorboard.name = conf.common.experiment_name

    conf.model.encoders.context_dim = conf.model.unet_params.context_dim
    return conf


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    return model

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    return model

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

def resize_mask_for_diffusion(mask):
    reduce_factor = max(1, (mask.size / 1024**2)**0.5)
    resized_mask = resize(
        mask,
        (
            (round(mask.shape[0] / reduce_factor) // 64) * 64,
            (round(mask.shape[1] / reduce_factor) // 64) * 64
        ),
        preserve_range=True,
        anti_aliasing=False
    )

    return resized_mask

def resize_image_for_diffusion(image):
    reduce_factor = max(1, (image.size[0] * image.size[1] / 1024**2)**0.5)
    image = image.resize((
        (round(image.size[0] / reduce_factor) // 64) * 64, (round(image.size[1] / reduce_factor) // 64) * 64
    ))

    return image

def get_polygon_mask_params(mask_size, box, num_vertices, mask_scale, min_scale, max_scale):
    center = ((box[2] + box[0]) / 2, (box[3] + box[1]) / 2)
    sizes = (box[2] - box[0], box[3] - box[1])

    part_avg_radii = np.linspace(mask_scale * sizes[0] / 2, mask_scale * sizes[1] / 2, num_vertices // 4)
    part_avg_radii = np.clip(part_avg_radii, min_scale * min(mask_size), max_scale * min(mask_size))
    avg_radii = np.concatenate([
        part_avg_radii,
        part_avg_radii[::-1],
        part_avg_radii,
        part_avg_radii[::-1],
    ])
    return center, avg_radii


def smooth_cerv(x, y):
    num_vertices = x.shape[0]
    x = np.concatenate((x[-3:-1], x, x[1:3]))
    y = np.concatenate((y[-3:-1], y, y[1:3]))
    t = np.arange(x.shape[0])

    ti = np.linspace(2, num_vertices + 1, 4 * num_vertices)
    xi = interp1d(t, x, kind='quadratic')(ti)
    yi = interp1d(t, y, kind='quadratic')(ti)
    return xi, yi


def get_polygon_mask(mask_size, mask_points):
    x, y = np.meshgrid(np.arange(mask_size[0]), np.arange(mask_size[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    path = Path(mask_points)
    grid = path.contains_points(points)
    grid = grid.reshape((mask_size[0], mask_size[1]))
    return 1. - grid.astype(np.int32)


def generate_polygon(mask_size, center, num_vertices, radii, radii_var, angle_var, smooth=True):
    angle_steps = np.random.uniform(1. - angle_var, 1. + angle_var, size=(num_vertices,))
    angle_steps = 2 * np.pi * angle_steps / angle_steps.sum()

    radii = np.random.normal(radii, radii_var * radii)
    radii = np.clip(radii, 0, 2 * radii)
    angles = np.cumsum(angle_steps)
    x = center[0] + radii * np.cos(angles)
    y = center[1] + radii * np.sin(angles)

    if smooth:
        x, y = smooth_cerv(x, y)
    points = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=-1)
    points = list(map(tuple, points.tolist()))
    return get_polygon_mask(mask_size, points)


def generate_circle_frame(mask_size, side_scales, num_vertices, radii_var, smooth=True):
    num_vertices_per_side = num_vertices // 4
    x_size, y_size = mask_size
    up_radii = np.array([y_size * (1. - side_scales[0]) // 2] * num_vertices_per_side)
    down_radii = np.array([y_size * (1. - side_scales[1]) // 2] * num_vertices_per_side)
    left_radii = np.array([x_size * (1. - side_scales[2]) // 2] * num_vertices_per_side)
    right_radii = np.array([x_size * (1. - side_scales[3]) // 2] * num_vertices_per_side)

    center = (x_size // 2, y_size // 2)
    radii = np.concatenate([
        right_radii[num_vertices_per_side // 2:],
        down_radii,
        left_radii,
        up_radii,
        right_radii[:num_vertices_per_side // 2],
    ])
    return 1. - generate_polygon(mask_size, center, num_vertices, radii, radii_var, 0., smooth=smooth)


def generate_square_frame(mask_size, side_scales, num_vertices, radii_var, smooth=True):
    num_vertices_per_side = num_vertices // 4
    x_size, y_size = mask_size
    diag_size = np.sqrt(x_size ** 2 + y_size ** 2)

    up_radii = np.linspace(
        diag_size * (1. - side_scales[0]) // 2,
        y_size * (1. - side_scales[0]) // 2,
        num_vertices_per_side // 2
    )
    down_radii = np.linspace(
        diag_size * (1. - side_scales[1]) // 2,
        y_size * (1. - side_scales[1]) // 2,
        num_vertices_per_side // 2
    )
    left_radii = np.linspace(
        diag_size * (1. - side_scales[2]) // 2,
        x_size * (1. - side_scales[2]) // 2,
        num_vertices_per_side // 2
    )
    right_radii = np.linspace(
        diag_size * (1. - side_scales[3]) // 2,
        x_size * (1. - side_scales[3]) // 2,
        num_vertices_per_side // 2
    )

    center = (x_size // 2, y_size // 2)
    radii = np.concatenate([
        right_radii[::-1],
        down_radii,
        down_radii[::-1],
        left_radii,
        left_radii[::-1],
        up_radii,
        up_radii[::-1],
        right_radii,
    ])
    return 1. - generate_polygon(mask_size, center, num_vertices, radii, radii_var, 0., smooth=smooth)

def generate_bbox(mask_size=(256, 256)):
    mask = np.zeros(mask_size)
    img_height = 256
    img_width = 256
    height, width = 32, 32
    ver_margin, hor_margin = 32, 32
    maxt = img_height - ver_margin - height
    maxl = img_width - hor_margin - width
    t = np.random.randint(low=ver_margin, high=maxt)
    l = np.random.randint(low=hor_margin, high=maxl)
    h = height
    w = width
    mask[t:t+h, l:l+w] = 1
    return  mask

def generate_mask(mask_size, box):
    mask = np.ones(mask_size)
    actions = np.random.randint(0, 2, (2,))
    if 0 in actions:
        if np.random.random() < 0.5:
            num_vertices = 16
            center, radii = get_polygon_mask_params(
                mask_size, box, num_vertices, mask_scale=1.5, min_scale=0.1, max_scale=0.6
            )
            mask *= generate_polygon(mask_size, center, num_vertices, radii, radii_var=0.15, angle_var=0.15)
        else:
            mask[int(box[0]): int(box[2]), int(box[1]): int(box[3])] = 0.
    if 1 in actions:
        radii_var = 0.15 * np.random.random()
        num_vertices = np.random.choice([16, 32])
        if np.random.random() < 0.5:
            if np.random.random() < 0.5:
                side_scales = 0.25 * np.random.random((4,)) + 0.05
                mask[:int(side_scales[0] * mask_size[0])] = 0.
                mask[-int(side_scales[1] * mask_size[0]):] = 0.
                mask[:, :int(side_scales[2] * mask_size[1])] = 0.
                mask[:, -int(side_scales[3] * mask_size[1]):] = 0.
            else:
                side_scales = 0.2 * np.random.random((4,)) + 0.05
                mask *= generate_square_frame(mask_size, side_scales, num_vertices, radii_var=radii_var)
        else:
            if np.random.random() < 0.5:
                side_scales = 0.15 * np.random.random((4,)) + 0.1
                mask[:int(side_scales[0] * mask_size[0])] = 0.
                mask[-int(side_scales[1] * mask_size[0]):] = 0.
                mask[:, :int(side_scales[2] * mask_size[1])] = 0.
                mask[:, -int(side_scales[3] * mask_size[1]):] = 0.
            else:
                side_scales = 0.15 * np.random.random((4,)) + 0.1
                mask *= generate_circle_frame(mask_size, side_scales, num_vertices, radii_var=radii_var)
    return mask
