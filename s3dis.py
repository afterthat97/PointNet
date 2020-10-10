import os
import glob
import numpy as np
import torch.utils.data


def get_block(room_xyz, room_rgb, room_gt, xyz_max, xcenter, ycenter, n_points):
    xmin, xmax = xcenter - 0.5, xcenter + 0.5
    ymin, ymax = ycenter - 0.5, ycenter + 0.5

    l, r = np.searchsorted(room_xyz[:, 0], [xmin, xmax])
    indices = np.where((room_xyz[l:r, 1] > ymin) & (room_xyz[l:r, 1] < ymax))[0] + l
    if len(indices) == 0:
        return indices, np.zeros([0, 9], dtype=np.float32), np.zeros([0, ], dtype=np.int64)
    if n_points != 'all':
        indices = np.random.choice(indices, n_points, indices.size < n_points)
    block_xyz, block_rgb, block_gt = room_xyz[indices, :], room_rgb[indices, :], room_gt[indices]

    # Each point is represented by a 9-dim vector of XYZ, RGB and normalized location
    # as to the room (from 0 to 1)
    block_data = np.zeros([len(indices), 9], dtype=np.float32)
    block_data[:, 0:3] = block_xyz[:, 0:3] - [xcenter, ycenter, 0]
    block_data[:, 3:6] = block_rgb / 255.0
    block_data[:, 6:9] = block_xyz / xyz_max
    block_data = np.transpose(block_data)  # [n_channels, n_points]

    return indices, block_data.astype(np.float32), block_gt.astype(np.int64)


class S3DIS(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, test_area, n_points=4096, sample_aug=1):
        super().__init__()

        assert os.path.isdir(dataset_dir)
        assert split == 'train' or split == 'test'
        assert type(test_area) == int and 1 <= test_area <= 6

        self.n_points = n_points
        self.rooms, self.indices = [], []

        for area_id in range(1, 7):
            if split == 'train' and area_id == test_area:
                continue
            if split == 'test' and area_id != test_area:
                continue
            area_dir = os.path.join(dataset_dir, 'Area_%d' % area_id)
            for room_path in glob.glob(os.path.join(area_dir, '*.npz')):
                room_data = np.load(room_path)
                room_area = np.ceil(np.amax(room_data['xyz'], axis=0))[:2].prod()
                self.indices.extend([len(self.rooms)] * int(room_area * sample_aug))
                self.rooms.append({
                    'xyz': room_data['xyz'],
                    'rgb': room_data['rgb'],
                    'gt': room_data['gt'],
                    'xyz_max': np.amax(room_data['xyz'], axis=0)
                })

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        room = self.rooms[self.indices[idx]]
        room_xyz, room_rgb, room_gt, xyz_max = room['xyz'], room['rgb'], room['gt'], room['xyz_max']
        xcenter, ycenter = room_xyz[np.random.choice(room_xyz.shape[0])][:2]
        _, block_data, block_gt = get_block(room_xyz, room_rgb, room_gt, xyz_max, xcenter, ycenter, self.n_points)
        return block_data, block_gt
