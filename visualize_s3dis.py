import yaml
import open3d
import logging
import numpy as np
from utils import init_logging
from omegaconf import DictConfig
from s3dis import S3DIS


def main():
    init_logging()

    with open('conf/dataset/s3dis.yaml') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    dataset = S3DIS(
        dataset_dir=cfgs.dataset.root_dir,
        split='test',
        test_area=cfgs.dataset.test_area,
        n_points=cfgs.dataset.n_points,
        block_type=cfgs.dataset.block_type,
        block_size=cfgs.dataset.block_size
    )

    logging.info('Dataset length: %d' % len(dataset))

    for pcloud, gt in dataset:
        pcloud = np.transpose(pcloud)
        pcloud_o3d = open3d.geometry.PointCloud()
        pcloud_o3d.points = open3d.utility.Vector3dVector(pcloud[:, 0:3])
        pcloud_o3d.colors = open3d.utility.Vector3dVector(pcloud[:, 3:6])
        open3d.visualization.draw_geometries([pcloud_o3d])


if __name__ == '__main__':
    main()
