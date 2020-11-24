import torch

try:
    from ._ball_query_cuda import _ball_query_cuda
    from ._furthest_point_sampling_cuda import _furthest_point_sampling_cuda
    from ._squared_distance_cuda import _squared_distance_cuda
except ImportError:
    raise ImportError('Failed to load one or more extensions')


def squared_distance(points_xyz_1, points_xyz_2, cpp_impl=True):
    """
    Calculate the Euclidean squared distance between every two points.
    :param points_xyz_1: the 1st set of points, [batch_size, n_points_1, 3]
    :param points_xyz_2: the 2nd set of points, [batch_size, n_points_2, 3]
    :param cpp_impl: whether to use the CUDA C++ implementation of squared-distance
    :return: squared distance between every two points, [batch_size, n_points_1, n_points_2]
    """
    @torch.cuda.amp.autocast(enabled=False)
    def _squared_distance_py(points_xyz_1, points_xyz_2):
        assert points_xyz_1.shape[0] == points_xyz_2.shape[0]
        batch_size, n_points1, n_points2 = points_xyz_1.shape[0], points_xyz_1.shape[1], points_xyz_2.shape[1]
        dist = -2 * torch.matmul(points_xyz_1, points_xyz_2.permute(0, 2, 1))
        dist += torch.sum(points_xyz_1 ** 2, -1).view(batch_size, n_points1, 1)
        dist += torch.sum(points_xyz_2 ** 2, -1).view(batch_size, 1, n_points2)
        return dist

    if cpp_impl:
        return _squared_distance_cuda(points_xyz_1, points_xyz_2)
    else:
        return _squared_distance_py(points_xyz_1, points_xyz_2)


def ball_query(points_xyz, centroids_xyz, radius, n_points_per_group, cpp_impl=True):
    """
    Finds all points that are within a radius to the centroids (an upper limit of n_points_per_group is set).
    :param points_xyz: a set of points, [batch_size, n_points, 3]
    :param centroids_xyz: a set of centroids, [batch_size, n_centroids, 3]
    :param radius: float
    :param n_points_per_group: int
    :param cpp_impl: whether to use the CUDA C++ implementation of ball-query
    :return: grouped indices of points, [batch_size, n_centroids, n_points_per_group]
    """
    def _ball_query_py(points_xyz, centroids_xyz, radius, n_points_per_group):
        batch_size, n_points, n_centroids = points_xyz.shape[0], points_xyz.shape[1], centroids_xyz.shape[1]
        indices = torch.arange(n_points, dtype=torch.long, device=points_xyz.device)
        grouped_indices = indices.view(1, 1, n_points).repeat([batch_size, n_centroids, 1])
        dists = squared_distance(centroids_xyz, points_xyz, cpp_impl=False)  # [batch_size, n_centroids, n_points]
        grouped_indices[dists >= radius ** 2] = n_points  # assign an out-of-range index
        grouped_indices = grouped_indices.sort(dim=-1)[0][:, :, :n_points_per_group]
        grouped_indices_first = grouped_indices[:, :, :1].expand(grouped_indices.shape)
        mask = grouped_indices == points_xyz.shape[1]
        grouped_indices[mask] = grouped_indices_first[mask]
        return grouped_indices

    if cpp_impl:
        grouped_indices = _ball_query_cuda(points_xyz, centroids_xyz, radius, n_points_per_group)
    else:
        grouped_indices = _ball_query_py(points_xyz, centroids_xyz, radius, n_points_per_group)

    return grouped_indices.to(torch.long)


def furthest_point_sampling(points_xyz, n_samples, cpp_impl=True):
    """
    :param points_xyz: a set of points, [batch_size, n_points, 3]
    :param n_samples: number of samples, int
    :param cpp_impl: whether to use the CUDA C++ implementation of furthest-point-sampling
    :return: indices of sampled points, [batch_size, n_samples]
    """
    def _furthest_point_sampling_py(points_xyz, n_samples):
        batch_size, n_points, _ = points_xyz.shape
        farthest_indices = torch.zeros(batch_size, n_samples, dtype=torch.long, device=points_xyz.device)
        distances = torch.ones(batch_size, n_points, device=points_xyz.device) * 1e10
        batch_indices = torch.arange(batch_size, dtype=torch.long, device=points_xyz.device)
        curr_farthest_idx = torch.zeros(batch_size, dtype=torch.long, device=points_xyz.device)
        for i in range(n_samples):
            farthest_indices[:, i] = curr_farthest_idx
            curr_farthest = points_xyz[batch_indices, curr_farthest_idx, :].view(batch_size, 1, 3)
            new_distances = torch.sum((points_xyz - curr_farthest) ** 2, -1)
            mask = new_distances < distances
            distances[mask] = new_distances[mask]
            curr_farthest_idx = torch.max(distances, -1)[1]
        return farthest_indices

    if cpp_impl:
        return _furthest_point_sampling_cuda(points_xyz, n_samples).to(torch.long)
    else:
        return _furthest_point_sampling_py(points_xyz, n_samples).to(torch.long)


def k_nearest_neighbor(points_xyz, centroids_xyz, k, cpp_impl=True):
    """
    :param points_xyz: a set of points, [batch_size, n_points, 3]
    :param centroids_xyz: a set of centroids, [batch_size, n_centroids, 3]
    :param k: int
    :param cpp_impl: whether to use the CUDA C++ implementation of k-nearest-neighbor
    :return: squared distances and indices of k-nearest neighbors, [batch_size, n_centroids, k]
    """
    dists = squared_distance(centroids_xyz, points_xyz, cpp_impl)  # [batch_size, n_centroids, n_points]
    topk = dists.topk(k, dim=-1, largest=False)
    return topk.values, topk.indices


def batch_indexing(batched_data, batched_indices):
    """
    :param batched_data: [batch_size, D, C1, ..., Cn]
    :param batched_indices: [batch_size, I1, I2, ..., Im]
    :return: indexed data: [batch_size, I1, I2, ..., Im, C1, C2, ..., Cn]
    """
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size = batched_data.shape[0]
    view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
    expand_shape = [batch_size] + list(batched_indices.shape)[1:]
    indices_of_batch = torch.arange(batch_size, dtype=torch.long, device=batched_data.device)
    indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)
    return batched_data[indices_of_batch, batched_indices, :]
