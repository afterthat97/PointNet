import os
import glob
import yaml
import open3d
import logging
import numpy as np
from utils import init_logging
from omegaconf import DictConfig


def load_room(room_path):
    room_data = np.load(room_path)
    return room_data['xyz'], room_data['rgb'] / 255.0


def main():
    init_logging()

    with open('conf/dataset/s3dis.yaml') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    area_dir = os.path.join(cfgs.root_dir, 'Area_1')
    for room_name in sorted(os.listdir(area_dir)):
        if not os.path.isdir(os.path.join(area_dir, room_name)):
            continue

        logging.info('Visualizing room %s...' % room_name)
        room1_xyz, room1_rgb = load_room(os.path.join(area_dir, '%s.npz' % room_name))
        room2_xyz, room2_rgb = load_room(os.path.join(area_dir, '%s_resampled.npz' % room_name))
        room2_xyz[:, 1] += np.max(room1_xyz[:, 1]) * 1.2

        pcloud_o3d = open3d.geometry.PointCloud()
        pcloud_o3d.points = open3d.utility.Vector3dVector(np.concatenate([room1_xyz, room2_xyz], axis=0))
        pcloud_o3d.colors = open3d.utility.Vector3dVector(np.concatenate([room1_rgb, room2_rgb], axis=0))
        open3d.visualization.draw_geometries([pcloud_o3d])


if __name__ == '__main__':
    main()
