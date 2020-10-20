import os
import glob
import yaml
import open3d
import logging
import numpy as np
from utils import init_logging, gen_random_colors
from omegaconf import DictConfig


def load_blocks(room_dir, offset_name):
    block_paths = glob.glob(os.path.join(room_dir, 'block_%s_*.npz' % offset_name))
    colors = gen_random_colors(len(block_paths))

    room_xyz, room_rgb = [], []
    for idx, block_path in enumerate(block_paths):
        data = np.load(block_path)
        room_xyz.append(data['block_xyz'])
        room_rgb.append(0.7 * data['block_rgb'] / 255.0 + 0.3 * colors[idx])

    return np.vstack(room_xyz), np.vstack(room_rgb)


def main():
    init_logging()

    with open('conf/dataset/s3dis.yaml') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    area_dir = os.path.join(cfgs.root_dir, 'Area_1')
    for room_name in sorted(os.listdir(area_dir)):
        room_dir = os.path.join(area_dir, room_name)
        if not os.path.isdir(room_dir):
            continue

        logging.info('Visualizing room %s...' % room_name)
        room1_xyz, room1_rgb = load_blocks(room_dir, 'zero')
        room2_xyz, room2_rgb = load_blocks(room_dir, 'half')
        room2_xyz[:, 1] += np.max(room1_xyz[:, 1]) * 1.2

        pcloud_o3d = open3d.geometry.PointCloud()
        pcloud_o3d.points = open3d.utility.Vector3dVector(np.concatenate([room1_xyz, room2_xyz], axis=0))
        pcloud_o3d.colors = open3d.utility.Vector3dVector(np.concatenate([room1_rgb, room2_rgb], axis=0))
        open3d.visualization.draw_geometries([pcloud_o3d])


if __name__ == '__main__':
    main()
