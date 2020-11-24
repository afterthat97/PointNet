#include "squared_distance.h"

void squared_distance_kernel_wrapper(const float* batched_points_xyz_1,
									 const float* batched_points_xyz_2,
									 int n_batch, int n_points_1, int n_points_2,
									 float* batched_dists);

torch::Tensor squared_distance_cuda(torch::Tensor points_xyz_1, torch::Tensor points_xyz_2) {
	bool swapped = false;
	if (points_xyz_1.size(1) < points_xyz_2.size(1)) {
		std::swap(points_xyz_1, points_xyz_2);
		swapped = true;
	}

	int64_t batch_size = points_xyz_1.size(0), n_points_1 = points_xyz_1.size(1), n_points_2 = points_xyz_2.size(1);
	torch::Tensor dists = torch::empty({batch_size, n_points_1, n_points_2}, torch::TensorOptions().device(points_xyz_1.device()));

	squared_distance_kernel_wrapper(points_xyz_1.data_ptr<float>(), points_xyz_2.data_ptr<float>(), batch_size, n_points_1, n_points_2, dists.data_ptr<float>());

	return swapped ? torch::transpose(dists, 1, 2) : dists;
}

#ifdef TORCH_EXTENSION_NAME
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("_squared_distance_cuda", &squared_distance_cuda, "CUDA implementation of squared-distance");
}
#endif