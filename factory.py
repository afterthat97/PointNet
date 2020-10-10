from omegaconf import DictConfig
from modelnet40 import ModelNet40
from s3dis import S3DIS
from models import PointNetCls, PointNetSeg


def dataset_factory(cfgs: DictConfig, split):
    assert split == 'train' or split == 'test'
    if cfgs.dataset.name == 'modelnet40':
        return ModelNet40(
            dataset_dir=cfgs.dataset.root_dir,
            split=split,
            n_points=cfgs.dataset.n_points
        )
    elif cfgs.dataset.name == 's3dis':
        return S3DIS(
            dataset_dir=cfgs.dataset.root_dir,
            split=split,
            test_area=cfgs.dataset.test_area,
            n_points=cfgs.dataset.n_points,
            sample_aug=cfgs.dataset.sample_aug[split]
        )
    else:
        raise NotImplementedError('Unknown dataset: %s' % cfgs.dataset.name)


def model_factory(cfgs: DictConfig):
    if cfgs.model.name == 'PointNetCls':
        assert cfgs.dataset.name == 'modelnet40'
        return PointNetCls(cfgs.dataset.n_classes)
    elif cfgs.model.name == 'PointNetSeg':
        assert cfgs.dataset.name == 's3dis'
        return PointNetSeg(cfgs.dataset.n_classes, cfgs.dataset.n_channels)
    else:
        raise NotImplementedError('Unknown model: %s' % cfgs.model.name)
