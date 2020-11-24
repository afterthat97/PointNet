import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_ext_modules():
    return [
        CUDAExtension(
            name='_ball_query_cuda',
            sources=[
                'ball_query/ball_query.cpp',
                'ball_query/ball_query_kernel.cu'
            ],
            include_dirs=['ball_query']
        ),
        CUDAExtension(
            name='_furthest_point_sampling_cuda',
            sources=[
                'furthest_point_sampling/furthest_point_sampling.cpp',
                'furthest_point_sampling/furthest_point_sampling_kernel.cu'
            ],
            include_dirs=['furthest_point_sampling']
        ),
        CUDAExtension(
            name='_squared_distance_cuda',
            sources=[
                'squared_distance/squared_distance.cpp',
                'squared_distance/squared_distance_kernel.cu'
            ],
            include_dirs=['squared_distance']
        )
    ]


os.environ['TORCH_CUDA_ARCH_LIST'] = '3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5'

setup(
    name='pointnet2_utils',
    ext_modules=get_ext_modules(),
    cmdclass={'build_ext': BuildExtension}
)
