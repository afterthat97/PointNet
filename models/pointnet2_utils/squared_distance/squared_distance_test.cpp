#include "squared_distance.h"
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

torch::Tensor squared_distance_std(torch::Tensor points_xyz_1, torch::Tensor points_xyz_2) {
	int64_t batch_size = points_xyz_1.size(0), n_points_1 = points_xyz_1.size(1), n_points_2 = points_xyz_2.size(1);
	torch::Tensor dists = -2 * torch::matmul(points_xyz_1, points_xyz_2.permute({0, 2, 1}));
	dists += torch::sum(points_xyz_1.pow(2), -1).view({batch_size, n_points_1, 1});
	dists += torch::sum(points_xyz_2.pow(2), -1).view({batch_size, 1, n_points_2});
	return dists;
}

int main() {
	constexpr int batch_size = 64;
	constexpr int n_points_1 = 4096;
	constexpr int n_points_2 = 1024;

	if (!torch::cuda::is_available()) {
		cout << "CUDA is not available, exiting..." << endl;
		return 1;
	}

	torch::manual_seed(0);
	torch::Tensor points_xyz_1 = torch::rand({batch_size, n_points_1, 3}), points_xyz_cuda_1 = points_xyz_1.cuda();
	torch::Tensor points_xyz_2 = torch::rand({batch_size, n_points_2, 3}), points_xyz_cuda_2 = points_xyz_2.cuda();

	// warm up...
	for (int t = 0; t < 3; t++) squared_distance_cuda(points_xyz_cuda_1, points_xyz_cuda_2);
	checkCudaErrors(cudaDeviceSynchronize());

	cout << "Running squared-distance on GPU... " << flush;
	auto t1 = chrono::high_resolution_clock::now();
	torch::Tensor dists_gpu = squared_distance_cuda(points_xyz_cuda_1, points_xyz_cuda_2);
	checkCudaErrors(cudaDeviceSynchronize());
	auto t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	cout << "Running squared-distance on CPU... " << flush;
	t1 = chrono::high_resolution_clock::now();
	torch::Tensor dists_std = squared_distance_std(points_xyz_1, points_xyz_2);
	t2 = chrono::high_resolution_clock::now();
	cout << "(" << chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() << "ms)" << endl;

	float diff = torch::mean(torch::abs(dists_std.cpu() - dists_gpu.cpu())).data_ptr<float>()[0];
	cout << "Checking results... " << (diff < 1e-6 ? "OK" : "Failed") << endl;

	return 0;
}