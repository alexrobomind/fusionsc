#pragma once

#include "common.h"

# define EIGEN_USE_THREADS 1
# define EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES 1

#ifdef FSC_WITH_CUDA
	#define EIGEN_USE_GPU
#endif

# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
# include <Eigen/Dense>
# include <Eigen/Core>
# include <cmath>

namespace fsc {
	using Eigen::Tensor;
	using Eigen::TensorFixedSize;
	using Eigen::TensorRef;
	using Eigen::TensorMap;
	using Eigen::Sizes;
	
	using Eigen::TensorOpCost;
}