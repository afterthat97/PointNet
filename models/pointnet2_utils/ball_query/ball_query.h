#pragma once

#include <torch/extension.h>

torch::Tensor ball_query_cuda(torch::Tensor points_xyz, torch::Tensor centroids_xyz, const float radius, const int n_points_per_group);
