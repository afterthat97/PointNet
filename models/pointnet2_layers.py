import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
from .pointnet2_utils import ball_query, furthest_point_sampling, k_nearest_neighbor, batch_indexing


def PointNet2SetAbstraction(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels):
    assert isinstance(n_samples, int)
    if isinstance(radius, ListConfig):  # multi-scale grouping
        assert n_samples > 1
        assert isinstance(n_points_per_group, ListConfig)
        assert isinstance(mlp_out_channels, ListConfig)
        assert len(radius) == len(n_points_per_group) == len(mlp_out_channels)
        return _PointNet2SetAbstractionMSG(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels)
    else:  # single-scale grouping
        return _PointNet2SetAbstractionSSG(n_samples, radius, n_points_per_group, in_channels, mlp_out_channels)


class _PointNet2SetAbstractionSSG(nn.Module):
    def __init__(self, n_samples, radius, n_points_per_group, in_channels, mlp_out_channels):
        super().__init__()
        self.n_samples = n_samples
        self.radius = radius
        self.n_points_per_group = n_points_per_group
        self.mlp_out_channels = mlp_out_channels
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        for out_channels in mlp_out_channels:
            self.mlp_convs.append(nn.Conv2d(in_channels, out_channels, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

    def forward(self, points_xyz, features=None):
        """
        :param points_xyz: 3D locations of points, [batch_size, n_points, 3]
        :param features: features of points, [batch_size, n_points, n_features]
        """
        batch_size, n_points = points_xyz.shape[0], points_xyz.shape[1]

        if features is None:
            features = torch.zeros([batch_size, n_points, 0], device=points_xyz.device)

        # sampling layer & grouping layer
        if self.n_samples > 1:
            # centroid_indices: [bs, n_samples]
            centroid_indices = furthest_point_sampling(points_xyz, self.n_samples)
            # centroids: [bs, n_samples, 3]
            centroids = batch_indexing(points_xyz, centroid_indices)
            # grouped_indices: [bs, n_samples, n_points_per_group]
            grouped_indices = ball_query(points_xyz, centroids, self.radius, self.n_points_per_group)
            # grouped_xyz: [bs, n_samples, n_points_per_group, 3]
            grouped_xyz = batch_indexing(points_xyz, grouped_indices)
            # grouped_features: [bs, n_samples, n_points_per_group, n_features]
            grouped_features = batch_indexing(features, grouped_indices)
            # grouped_xyz_norm: [bs, n_samples, n_points_per_group, 3]
            grouped_xyz_norm = grouped_xyz - centroids.view(batch_size, self.n_samples, 1, 3)
            # grouped_features: [bs, n_samples, n_points_per_group, n_features + 3]
            grouped_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        else:
            centroids = torch.zeros([batch_size, 1, 3], dtype=points_xyz.dtype, device=points_xyz.device)
            grouped_xyz = points_xyz.view(batch_size, 1, n_points, 3)
            grouped_features = torch.cat([grouped_xyz, features.view(batch_size, 1, n_points, -1)], dim=-1)

        # PointNet layer: MLP (by 1x1 Conv2d), BN, and max-pooling
        grouped_features = grouped_features.transpose(1, 3)  # [bs, n_features + 3, n_points_per_group, n_samples]
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            grouped_features = F.relu(bn(conv(grouped_features)))

        grouped_features = torch.max(grouped_features, 2)[0]  # [bs, n_mlp_channels, n_samples]
        grouped_features = grouped_features.transpose(1, 2)  # [bs, n_samples, n_mlp_channels]

        return centroids, grouped_features

    def out_channels(self):
        return self.mlp_out_channels[-1]


class _PointNet2SetAbstractionMSG(nn.Module):
    def __init__(self, n_samples, radius_list, n_points_per_group_list, in_channels, mlp_out_channels_list):
        super().__init__()

        self.n_samples = n_samples
        self.radius_list = radius_list
        self.n_points_per_group_list = n_points_per_group_list
        self.mlp_out_channels_list = mlp_out_channels_list
        self.mlp_convs_list = nn.ModuleList()
        self.mlp_bns_list = nn.ModuleList()

        for mlp_out_channels in mlp_out_channels_list:
            mlp_convs, mlp_bns = nn.ModuleList(), nn.ModuleList()
            last_channels = in_channels
            for out_channels in mlp_out_channels:
                mlp_convs.append(nn.Conv2d(last_channels, out_channels, 1))
                mlp_bns.append(nn.BatchNorm2d(out_channels))
                last_channels = out_channels
            self.mlp_convs_list.append(mlp_convs)
            self.mlp_bns_list.append(mlp_bns)

    def forward(self, points_xyz, features=None):
        """
        :param points_xyz: 3D locations of points, [batch_size, n_points, 3]
        :param features: features of points, [batch_size, n_points, n_features]
        """
        batch_size, n_points = points_xyz.shape[0], points_xyz.shape[2]
        if features is None:
            features = torch.zeros([batch_size, n_points, 0], device=points_xyz.device)

        # centroid_indices: [bs, n_samples]
        centroid_indices = furthest_point_sampling(points_xyz, self.n_samples)
        # centroids: [bs, n_samples, 3]
        centroids = batch_indexing(points_xyz, centroid_indices)

        multi_scale_features = []
        for radius, n_points_per_group, mlp_convs, mlp_bns in zip(
                self.radius_list, self.n_points_per_group_list, self.mlp_convs_list, self.mlp_bns_list):
            # grouped_indices: [bs, n_samples, n_points_per_group]
            grouped_indices = ball_query(points_xyz, centroids, radius, n_points_per_group)
            # grouped_xyz: [bs, n_samples, n_points_per_group, 3]
            grouped_xyz = batch_indexing(points_xyz, grouped_indices)
            # grouped_features: [bs, n_samples, n_points_per_group, n_features]
            grouped_features = batch_indexing(features, grouped_indices)
            # grouped_xyz_norm: [bs, n_samples, n_points_per_group, 3]
            grouped_xyz_norm = grouped_xyz - centroids.view(batch_size, self.n_samples, 1, 3)
            # grouped_features: [bs, n_samples, n_points_per_group, n_features + 3]
            grouped_features = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)

            # PointNet layer: MLP (by 1x1 Conv2d), BN, and max-pooling
            grouped_features = grouped_features.transpose(1, 3)  # [bs, n_features + 3, n_points_per_group, n_samples]
            for conv, bn in zip(mlp_convs, mlp_bns):
                grouped_features = F.relu(bn(conv(grouped_features)))
            grouped_features = torch.max(grouped_features, 2)[0]  # [bs, n_mlp_channels, n_samples]
            grouped_features = grouped_features.transpose(1, 2)  # [bs, n_samples, n_mlp_channels]

            multi_scale_features.append(grouped_features)

        multi_scale_features = torch.cat(multi_scale_features, dim=-1)

        return centroids, multi_scale_features

    def out_channels(self):
        return sum(mlp_out_channels[-1] for mlp_out_channels in self.mlp_out_channels_list)


class PointNet2FeaturePropagation(nn.Module):
    def __init__(self, in_channels, mlp_out_channels):
        super().__init__()
        self.mlp_out_channels = mlp_out_channels
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()

        for out_channels in mlp_out_channels:
            self.mlp_convs.append(nn.Conv1d(in_channels, out_channels, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channels))
            in_channels = out_channels

    def forward(self, sampled_xyz, sampled_features, original_xyz, original_features, k):
        """
        :param sampled_xyz: 3D locations of sampled points, [batch_size, n_samples, 3]
        :param sampled_features: features of sampled points, [batch_size, n_samples, n_sampled_features]
        :param original_xyz: 3D locations of original points, [batch_size, n_original, 3]
        :param original_features: features of original points, [batch_size, n_original, n_original_features]
        :param k: k-nearest neighbor, int
        """
        knn_dists, knn_indices = k_nearest_neighbor(sampled_xyz, original_xyz, k)  # [bs, n_original, k]
        knn_weights = 1.0 / (knn_dists + 1e-8)  # [bs, n_original, k]
        knn_weights = knn_weights / torch.sum(knn_weights, dim=-1, keepdim=True)  # [bs, n_original, k]

        # knn_features: [bs, n_original, k, n_sampled_features]
        knn_features = batch_indexing(sampled_features, knn_indices)
        # interpolated_features: [bs, n_original, n_sampled_features]
        interpolated_features = torch.sum(knn_features * knn_weights[:, :, :, None], dim=2)
        # concatenated_features: [bs, n_original, n_original_features + n_sampled_features]
        concatenated_features = torch.cat([original_features, interpolated_features], dim=-1)

        # PointNet layer: MLP (by 1x1 Conv2d), BN, and max-pooling
        concatenated_features = concatenated_features.transpose(1, 2)
        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            concatenated_features = F.relu(bn(conv(concatenated_features)))
        concatenated_features = concatenated_features.transpose(1, 2)

        return concatenated_features

    def out_channels(self):
        return self.mlp_out_channels[-1]
