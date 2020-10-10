import yaml
import open3d
import logging
import numpy as np
from utils import init_logging
from omegaconf import DictConfig
from modelnet40 import ModelNet40


def main():
    init_logging()

    with open('conf/dataset/modelnet40.yaml') as f:
        cfgs = DictConfig(yaml.load(f, Loader=yaml.FullLoader))

    dataset = ModelNet40(
        dataset_dir=cfgs.root_dir,
        split='train',
        n_points=cfgs.n_points
    )

    for xyz, class_id in dataset:
        logging.info('class name: %s' % dataset.class_id2name(class_id))
        xyz = np.transpose(xyz)
        pcloud_o3d = open3d.geometry.PointCloud()
        pcloud_o3d.points = open3d.utility.Vector3dVector(xyz)
        open3d.visualization.draw_geometries([pcloud_o3d])


if __name__ == '__main__':
    main()
