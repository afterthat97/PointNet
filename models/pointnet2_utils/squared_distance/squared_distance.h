#pragma once

#include <torch/extension.h>

torch::Tensor squared_distance_cuda(torch::Tensor points_xyz_1, torch::Tensor points_xyz_2);