import torch
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
            n_points=cfgs.dataset.n_points,
            random_rotate=cfgs.dataset.random_rotate,
            random_jitter=cfgs.dataset.random_jitter
        )
    elif cfgs.dataset.name == 's3dis':
        return S3DIS(
            dataset_dir=cfgs.dataset.root_dir,
            split=split,
            test_area=cfgs.dataset.test_area,
            n_points=cfgs.dataset.n_points,
            max_dropout=cfgs.dataset.max_dropout,
            block_type=cfgs.dataset.block_type,
            block_size=cfgs.dataset.block_size
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


def optimizer_factory(cfgs: DictConfig, params):
    if cfgs.training.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            params=params,
            lr=cfgs.training.lr.init_value,
            weight_decay=cfgs.training.weight_decay
        )
    elif cfgs.training.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            params=params,
            lr=cfgs.training.lr.init_value,
            weight_decay=cfgs.training.weight_decay
        )
    elif cfgs.training.optimizer == 'adadelta':
        optimizer = torch.optim.Adadelta(
            params=params,
            lr=cfgs.training.lr.init_value,
            weight_decay=cfgs.training.weight_decay
        )
    elif cfgs.training.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            params=params,
            lr=cfgs.training.lr.init_value,
            momentum=cfgs.training.lr.momentum,
            weight_decay=cfgs.training.weight_decay
        )
    elif cfgs.training.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            params=params,
            lr=cfgs.training.lr.init_value,
            momentum=cfgs.training.lr.momentum,
            weight_decay=cfgs.training.weight_decay
        )
    else:
        raise NotImplementedError('Unknown optimizer: %s' % cfgs.training.optimizer)

    if isinstance(cfgs.training.lr.decay_milestones, int):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=cfgs.training.lr.decay_milestones,
            gamma=cfgs.training.lr.decay_rate
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfgs.training.lr.decay_milestones,
            gamma=cfgs.training.lr.decay_rate
        )

    return optimizer, lr_scheduler
