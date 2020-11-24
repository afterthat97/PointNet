import torch
from unittest import TestCase


class Test(TestCase):
    def test_ball_query(self):
        from models.pointnet2_utils import ball_query

        batch_size, n_points, n_centroids, radius, n_points_per_group = 32, 100, 10, 0.2, 10
        points_xyz = torch.rand([batch_size, n_points, 3]).cuda()
        centroids_xyz = torch.rand([batch_size, n_centroids, 3]).cuda()

        indices_cpp = ball_query(points_xyz, centroids_xyz, radius, n_points_per_group, cpp_impl=True)
        indices_std = ball_query(points_xyz, centroids_xyz, radius, n_points_per_group, cpp_impl=False)
        torch.cuda.synchronize()

        for b in range(indices_cpp.shape[0]):
            for i in range(indices_cpp.shape[1]):
                unique_indices_cpp = torch.unique_consecutive(indices_cpp[b][i].sort()[0])
                unique_indices_std = torch.unique_consecutive(indices_std[b][i].sort()[0])
                assert torch.equal(unique_indices_cpp, unique_indices_std)

    def test_furthest_point_sampling(self):
        from models.pointnet2_utils import furthest_point_sampling

        batch_size, n_points, n_samples = 32, 100, 10
        points_xyz = torch.rand([batch_size, n_points, 3]).cuda()

        indices_cpp = furthest_point_sampling(points_xyz, n_samples, cpp_impl=True)
        indices_std = furthest_point_sampling(points_xyz, n_samples, cpp_impl=False)
        torch.cuda.synchronize()

        assert torch.equal(indices_cpp, indices_std)

    def test_squared_distance(self):
        from models.pointnet2_utils import squared_distance

        batch_size, n_points_1, n_points_2 = 32, 100, 10
        points_xyz_1 = torch.rand([batch_size, n_points_1, 3]).cuda()
        points_xyz_2 = torch.rand([batch_size, n_points_2, 3]).cuda()

        dists_cpp = squared_distance(points_xyz_1, points_xyz_2, cpp_impl=True)
        dists_std = squared_distance(points_xyz_1, points_xyz_2, cpp_impl=False)
        torch.cuda.synchronize()

        assert torch.mean(dists_cpp - dists_std) < 1e-6

    def test_k_nearest_neighbor(self):
        from models.pointnet2_utils import k_nearest_neighbor

        batch_size, n_points, n_centroids, k = 32, 100, 10, 3
        points_xyz = torch.rand([batch_size, n_points, 3]).cuda()
        centroids_xyz = torch.rand([batch_size, n_centroids, 3]).cuda()

        dists_cpp, indices_cpp = k_nearest_neighbor(points_xyz, centroids_xyz, k, cpp_impl=True)
        dists_std, indices_std = k_nearest_neighbor(points_xyz, centroids_xyz, k, cpp_impl=False)
        torch.cuda.synchronize()

        assert torch.mean(dists_cpp - dists_std) < 1e-6
        assert torch.equal(indices_cpp, indices_std)
