import sys
import math
import open3d
import random
import logging
import colorsys
import numpy as np
import torch.utils.data
import torch.distributed as dist


class _RepeatSampler(object):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


def init_logging(filename=None):
    logging.root = logging.RootLogger('INFO')
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s')

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)


def save_ply(filepath, xyz, rgb):
    pcloud_o3d = open3d.geometry.PointCloud()
    pcloud_o3d.points = open3d.utility.Vector3dVector(xyz)
    pcloud_o3d.colors = open3d.utility.Vector3dVector(rgb)
    open3d.io.write_point_cloud(filepath, pcloud_o3d)


def argsplit(n_elements_tot, n_elements_per_split):
    rand_indices = np.arange(n_elements_tot)
    random.shuffle(rand_indices)

    n_split = math.ceil(n_elements_tot / n_elements_per_split)
    n_points_avg = math.ceil(n_elements_tot / n_split)
    n_points = [n_points_avg] * n_split
    n_points[-1] = n_elements_tot - n_points_avg * (n_split - 1)
    starts = [0] + list(np.cumsum(n_points))

    indices = np.zeros([n_split, n_elements_per_split], int)
    for idx in range(n_split):
        start, end = starts[idx], starts[idx] + n_points[idx]
        repeat_times = math.ceil(n_elements_per_split / n_points[idx])
        repeated_indices = np.tile(rand_indices[start:end], repeat_times)
        indices[idx] = repeated_indices[:n_elements_per_split]

    return indices


def gen_random_colors(num, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / num, 1, brightness) for i in range(num)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [np.array(color, np.float32) for color in colors]
    random.shuffle(colors)
    return colors


def dist_reduce_sum(value, n_gpus):
    if n_gpus == 1:
        return value
    tensor = torch.Tensor([value]).cuda()
    dist.all_reduce(tensor)
    return tensor.item()


def get_ious(pred, target, n_classes):
    ious = []
    for i in range(n_classes):
        intersection = ((target == i) & (pred == i)).sum()
        union = (target == i).sum() + (pred == i).sum() - intersection
        ious.append(100.0 * intersection / union if union > 0 else np.nan)
    return ious
