import time
import torch


def bench_ball_query():
    from models.pointnet2_utils import ball_query

    batch_size, n_points, n_centroids, radius, n_points_per_group = 64, 4096, 1024, 0.5, 64
    points_xyz = torch.rand([batch_size, n_points, 3]).cuda()
    centroids_xyz = torch.rand([batch_size, n_centroids, 3]).cuda()

    # warm up
    for _ in range(3):
        ball_query(points_xyz, centroids_xyz, radius, n_points_per_group)
        torch.cuda.synchronize()

    start_time = time.time_ns()
    for _ in range(10):
        ball_query(points_xyz, centroids_xyz, radius, n_points_per_group)
        torch.cuda.synchronize()
    timing = (time.time_ns() - start_time) / 10

    print('Timing for ball-query: %.2fms' % (timing / 1e6))


def bench_furthest_point_sampling():
    from models.pointnet2_utils import furthest_point_sampling

    batch_size, n_points, n_samples = 64, 4096, 1024
    points_xyz = torch.rand([batch_size, n_points, 3]).cuda()

    # warm up
    for _ in range(3):
        furthest_point_sampling(points_xyz, n_samples)
        torch.cuda.synchronize()

    start_time = time.time_ns()
    for _ in range(10):
        furthest_point_sampling(points_xyz, n_samples)
        torch.cuda.synchronize()
    timing = (time.time_ns() - start_time) / 10

    print('Timing for furthest-point-sampling: %.2fms' % (timing / 1e6))


def bench_squared_distance():
    from models.pointnet2_utils import squared_distance

    batch_size, n_points_1, n_points_2 = 64, 4096, 1024
    points_xyz_1 = torch.rand([batch_size, n_points_1, 3]).cuda()
    points_xyz_2 = torch.rand([batch_size, n_points_2, 3]).cuda()

    # warm up
    for _ in range(3):
        squared_distance(points_xyz_1, points_xyz_2)
        torch.cuda.synchronize()

    start_time = time.time_ns()
    for _ in range(10):
        squared_distance(points_xyz_1, points_xyz_2)
        torch.cuda.synchronize()
    timing = (time.time_ns() - start_time) / 10

    print('Timing for squared-distance: %.2fms' % (timing / 1e6))


def bench_k_nearest_neighbor():
    from models.pointnet2_utils import k_nearest_neighbor

    batch_size, n_points, n_centroids, k = 64, 4096, 1024, 3
    points_xyz = torch.rand([batch_size, n_points, 3]).cuda()
    centroids_xyz = torch.rand([batch_size, n_centroids, 3]).cuda()

    # warm up
    for _ in range(3):
        k_nearest_neighbor(points_xyz, centroids_xyz, k)
        torch.cuda.synchronize()

    start_time = time.time_ns()
    for _ in range(10):
        k_nearest_neighbor(points_xyz, centroids_xyz, k)
        torch.cuda.synchronize()
    timing = (time.time_ns() - start_time) / 10

    print('Timing for k-nearest-neighbor: %.2fms' % (timing / 1e6))


if __name__ == '__main__':
    bench_ball_query()
    bench_furthest_point_sampling()
    bench_squared_distance()
    bench_k_nearest_neighbor()
