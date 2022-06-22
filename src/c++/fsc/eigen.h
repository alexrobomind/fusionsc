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

# undef NOCOPY
# undef IN
# undef OUT

namespace fsc {
	using Eigen::Tensor;
	using Eigen::TensorFixedSize;
	using Eigen::TensorRef;
	using Eigen::TensorMap;
	using Eigen::Sizes;
	
	using Eigen::TensorOpCost;
	
	constexpr inline double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp

	template<typename T, unsigned int n>
	using TVec = Eigen::TensorFixedSize<T, Eigen::Sizes<n>>;

	template<typename T>
	using TVec3 = TVec<T, 3>;

	template<typename T>
	using TVec4 = TVec<T, 4>;

	using TVec3d = TVec3<double>;
	using TVec4d = TVec4<double>;

	template<typename T, unsigned int n>
	using Vec = Eigen::Vector<T, n>;

	template<typename T>
	using Vec2 = Vec<T, 2>;

	template<typename T>
	using Vec3 = Vec<T, 3>;

	template<typename T>
	using Vec4 = Vec<T, 4>;

	using Vec2d = Vec<double, 2>;
	using Vec3d = Vec<double, 3>;
	using Vec4d = Vec<double, 4>;

	using Vec2f = Vec<float, 2>;
	using Vec3f = Vec<float, 3>;
	using Vec4f = Vec<float, 4>;

	using Vec2u = Vec<unsigned int, 2>;
	using Vec3u = Vec<unsigned int, 3>;
	using Vec4u = Vec<unsigned int, 4>;

	using Vec2i = Vec<int, 2>;
	using Vec3i = Vec<int, 3>;
	using Vec4i = Vec<int, 4>;

	template<typename T>
	using Mat4 = Eigen::Matrix<T, 4, 4>;

	template<typename T>
	using Mat3 = Eigen::Matrix<T, 3, 3>;

	using Mat4d = Mat4<double>;
	using Mat3d = Mat3<double>;
}