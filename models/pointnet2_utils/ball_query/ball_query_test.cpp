#include "ball_query.h"
#include <chrono>
#include <cuda_runtime.h>
using namespace std;

void _checkCudaErrors(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
				static_cast<unsigned int>(result), cudaGetErrorName(result), func);
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}
#define checkCudaErrors(val) _checkCudaErrors((val), #val, __FILE__, __LINE__)

torch::Tensor ball_query_std(torch::Tensor points_xyz, torch::Tensor centroids_xyz, const float radius, const int n_points_per_group) {
	int64_t batch_size = points_xyz.size(0), n_points = points_xyz.size(1), n_centroids = centroids_xyz.size(1);
	torch::Tensor indices = torch::ones({batch_size, n_centroids, n_points_per_group}, torch::TensorOptions().dtype(torch::kInt64)) * n_points;

	float radius2 = radius * radius;
	for (int b = 0; b < batch_size; b++)
		for (int i = 0; i < n_centroids; i++) {
			torch::Tensor dists = torch::sum((points_xyz[b] - centroids_xyz[b][i]).pow(2), -1);
			indices[b][i] = -std::get<0>((-torch::arange(n_points).index_put_({dists > radius2}, n_points)).topk(n_points_per_group));
			indices[b][i] = indices[b][i].index_put_({indices[b][i] == n_points}, indices[b][i].data_ptr<int64_t>()[0]);
		}

	return indices;
}

bool check_result(torch::Tensor indices1, torch::Tensor indices2) {
	int64_t batch_size = indices1.size(0), n_centroids = indices1.size(1);

	for (int b = 0; b < batch_size; b++)
		for (int i = 0; i < n_centroids; i++) {
			torch::Tensor unique_indices1 = std::get<0>(torch::unique_consecutive(std::get<0>(indices1[b][i].sort())));
			torch::Tensor unique_indices2 = std::get<0>(torch::unique_consecutive(std::get<0>(indices2[b][i].sort())));
			if (!torch::equal(unique_indices1, unique_indices2)) return false;
		}

	return true;
}

int main() {
	constexpr int batch_size = 32;
	constexpr int n_points = 1024;
	constexpr int n_centroids = 128;
	constexpr float radius = 0.1f;
	constexpr int n_points_per_group = 32;

	if (!torch::cuda::is_available()) {
		cout << "CUDA is not available, exiting..." << endl;
		return 1;
	}

	torch::manual_seed(0);
	torch::Tensor points_xyz = torch::rand({batch_size, n_points, 3}), points_xyz_cuda = points_xyz.cuda();
	torch::Tensor centroids_xyz = torch::rand({batch_size, n_centroids, 3}), centroids_xyz_cuda = centroids_xyz.cuda();

	// warm up...
	for (int t = 0; t < 3; t++) ball_query_cuda(points_xyz_cuda, centroids_xyz_cuda, radius, n_points_per_group);
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "Running ball query on GPU... " << flush;
	auto t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_gpu = ball_query_cuda(points_xyz_cuda, centroids_xyz_cuda, radius, n_points_per_group);
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Running ball query on CPU... " << flush;
	t1 = chrono::high_resolution_clock::now();
	torch::Tensor indices_std = ball_query_std(points_xyz, centroids_xyz, radius, n_points_per_group);
	t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Checking results... " << (check_result(indices_std.cpu(), indices_gpu.cpu()) ? "OK" : "Failed") << endl;
}