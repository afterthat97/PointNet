import os
import numpy as np
import torch.utils.data


def normalize_pcloud(pcloud):
    centroid = np.mean(pcloud, axis=0)
    pcloud = pcloud - centroid
    pcloud = pcloud / np.max(np.sqrt(np.sum(pcloud ** 2, axis=1)))
    return pcloud


def random_rotate_pcloud(pcloud):
    rot_angle = np.random.uniform() * 2 * np.pi
    sin, cos = np.sin(rot_angle), np.cos(rot_angle)
    rot_matrix = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    pcloud = np.dot(pcloud, rot_matrix)
    return pcloud


def random_jitter_pcloud(pcloud, sigma=0.01):
    return pcloud + sigma * np.random.randn(*pcloud.shape)


class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, split, n_points):
        assert os.path.isdir(dataset_dir)
        assert split == 'train' or split == 'test'
        self.dataset_dir = dataset_dir
        self.split = split
        self.n_points = n_points
        self.class_dict = self._load_class_dict()
        self.model_indices = self._load_model_indices()

    def __len__(self):
        return len(self.model_indices)

    def __getitem__(self, index):
        return self._load_model(self.model_indices[index])

    def class_id2name(self, class_id):
        return list(self.class_dict.keys())[class_id]

    def class_name2id(self, class_name):
        return self.class_dict[class_name]

    def _load_class_dict(self):
        filepath = os.path.join(self.dataset_dir, 'modelnet40_shape_names.txt')
        assert os.path.exists(filepath)
        class_names = [line.rstrip() for line in open(filepath)]
        return dict(zip(class_names, range(len(class_names))))

    def _load_model_indices(self):
        filepath = os.path.join(self.dataset_dir, 'modelnet40_%s.txt' % self.split)
        assert os.path.exists(filepath)
        return [line.rstrip() for line in open(filepath)]

    def _load_model(self, model_idx):
        class_name = '_'.join(model_idx.split('_')[0:-1])
        class_id = self.class_name2id(class_name)
        filepath = os.path.join(self.dataset_dir, class_name, model_idx + '.npz')
        assert os.path.exists(filepath)

        xyz = np.load(filepath)['xyz']

        if self.split == 'train':
            indices = np.random.choice(xyz.shape[0], self.n_points)
            xyz = xyz[indices, :]
            xyz = random_rotate_pcloud(xyz)
            xyz = random_jitter_pcloud(xyz)
        else:
            xyz = xyz[:self.n_points, :]

        xyz = normalize_pcloud(xyz)
        xyz = np.transpose(xyz)  # [n_channels, n_points]

        return xyz.astype(np.float32), class_id
