import os
import math
import glob
import numpy as np
import torch.utils.data


def prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max):
    """
    Each point is represented by a 9-dim vector of XYZ, RGB
    and normalized location as to the room (from 0 to 1)
    """
    block_data = np.zeros([len(block_xyz), 9], dtype=np.float32)
    block_data[:, 0:3] = block_xyz[:, 0:3] - [xcenter, ycenter, 0]
    block_data[:, 3:6] = block_rgb / 255.0
    block_data[:, 6:9] = block_xyz / room_xyz_max
    return block_data


class _S3disStatic(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, area_ids, n_points=4096, offset_name=''):
        super().__init__()
        self.n_points = n_points
        self.block_paths = []

        for area_id in area_ids:
            area_dir = os.path.join(dataset_dir, 'Area_%d' % area_id)
            for room_name in sorted(os.listdir(area_dir)):
                room_dir = os.path.join(area_dir, room_name)
                for block_path in glob.glob(os.path.join(room_dir, 'block_%s*.npz' % offset_name)):
                    n_points_in_block = np.load(block_path)['n_points_in_block']
                    self.block_paths.extend([block_path] * math.ceil(n_points_in_block / self.n_points))

    def __len__(self):
        return len(self.block_paths)

    def __getitem__(self, index):
        data = np.load(self.block_paths[index])
        block_xyz, block_rgb, block_gt = data['block_xyz'], data['block_rgb'], data['block_gt']
        block_size, room_xyz_max = data['block_size'], data['room_xyz_max']

        xcenter, ycenter = np.amin(block_xyz, axis=0)[:2] + block_size / 2
        block_data = prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)

        choices = np.random.choice(len(block_xyz), self.n_points, len(block_xyz) < self.n_points)
        block_data = block_data[choices].transpose()  # [n_channels, n_points]
        block_gt = block_gt[choices]

        return block_data.astype(np.float32), block_gt.astype(np.int64)


class _S3disDynamic(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, area_ids, n_points=4096, block_size=1.0, sample_aug=1):
        super().__init__()
        self.n_points, self.block_size = n_points, block_size
        self.rooms, self.indices = [], []

        for area_id in area_ids:
            area_dir = os.path.join(dataset_dir, 'Area_%d' % area_id)
            for room_path in glob.glob(os.path.join(area_dir, '*_resampled.npz')):
                room_data = np.load(room_path)
                self.indices.extend([len(self.rooms)] * math.ceil(room_data['n_points'] / self.n_points) * sample_aug)
                self.rooms.append({
                    'xyz': room_data['xyz'],
                    'rgb': room_data['rgb'],
                    'gt': room_data['gt'],
                    'xyz_max': np.amax(room_data['xyz'], axis=0)
                })

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        room = self.rooms[self.indices[index]]
        room_xyz, room_rgb, room_gt, room_xyz_max = room['xyz'], room['rgb'], room['gt'], room['xyz_max']

        xcenter, ycenter = room_xyz[np.random.choice(room_xyz.shape[0])][:2]
        indices = self._get_block_indices(room_xyz, xcenter, ycenter)
        block_xyz, block_rgb, block_gt = room_xyz[indices], room_rgb[indices], room_gt[indices]

        block_data = prepare_input(block_xyz, block_rgb, xcenter, ycenter, room_xyz_max)
        block_data = np.transpose(block_data)  # [n_channels, n_points]

        return block_data.astype(np.float32), block_gt.astype(np.int64)

    def _get_block_indices(self, room_xyz, xcenter, ycenter):
        xmin, xmax = xcenter - self.block_size / 2, xcenter + self.block_size / 2,
        ymin, ymax = ycenter - self.block_size / 2, ycenter + self.block_size / 2
        l, r = np.searchsorted(room_xyz[:, 0], [xmin, xmax])
        indices = np.where((room_xyz[l:r, 1] > ymin) & (room_xyz[l:r, 1] < ymax))[0] + l
        if len(indices) == 0:
            return indices, np.zeros([0, 9], dtype=np.float32), np.zeros([0, ], dtype=np.int64)
        if self.n_points != 'all':
            indices = np.random.choice(indices, self.n_points, indices.size < self.n_points)
        return indices


class S3DIS(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, test_area, n_points=4096, block_type='dynamic', block_size=1.0):
        super().__init__()

        assert os.path.isdir(dataset_dir)
        assert split == 'train' or split == 'test'
        assert type(test_area) == int and 1 <= test_area <= 6

        area_ids = []
        for area_id in range(1, 7):
            if split == 'train' and area_id == test_area:
                continue
            if split == 'test' and area_id != test_area:
                continue
            area_ids.append(area_id)

        if block_type == 'static':
            offset_name = 'zero' if split == 'test' else ''
            self.dataset = _S3disStatic(dataset_dir, area_ids, n_points, offset_name)
        elif block_type == 'dynamic':
            sample_aug = 1 if split == 'test' else 2
            self.dataset = _S3disDynamic(dataset_dir, area_ids, n_points, block_size, sample_aug)
        else:
            raise NotImplementedError('Unknown block type: %s' % block_type)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]
