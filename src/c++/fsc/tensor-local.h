#pragma once

# define EIGEN_USE_THREADS 1
# define EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES 1

#ifdef __CUDACC__
	#define EIGEN_USE_GPU
#endif

#ifdef HIP
	#define EIGEN_USE_GPU
#endif

# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
# include <Eigen/Dense>
# include <cmath>

namespace fsc {
	
using Eigen::Tensor;
using Eigen::TensorFixedSize;
using Eigen::TensorRef;
using Eigen::TensorMap;
using Eigen::Sizes;

constexpr double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp

template<typename T>
using Vec3 = TensorFixedSize<T, Sizes<3>>;

using Vec3d = Vec3<double>;
using Vec3f = Vec3<float>;


template<typename T>
typename T::Scalar normSq(const T& t) { TensorFixedSize<typename T::Scalar, Sizes<>> result = t.square().sum(); return result(); }

template<typename T>
Vec3<T> EIGEN_DEVICE_FUNC cross(const Vec3<T>& t1, const Vec3<T>& t2);

template<typename T, unsigned int dim>
T EIGEN_DEVICE_FUNC normSq(const TensorFixedSize<T, Sizes<dim>>& t) {
	T result = 0;
	for(unsigned int i = 0; i < dim; ++i)
		result += t[i] * t[i];
	return result;
}

template<typename T, unsigned int dim>
T EIGEN_DEVICE_FUNC norm(const TensorFixedSize<T, Sizes<dim>>& t) {
	return sqrt(normSq(t));
}

/**
 * Helper struct that can be used to map data towards a specific device for calculation. Is constructed with a host pointer and a size,
 * and allocates a corresponding device pointer.
 */
template<typename T, typename Device>
struct MappedData;

/**
 * Tensor constructed on a device sharing a host tensor. Subclasses Eigen::TensorRef<T>.
 */
template<typename T, typename Device>
struct MappedTensor { static_assert(sizeof(T) == 0, "Mapper not implemented"); };

template<typename TVal, int rank, int options, typename Index, typename Device>
struct MappedTensor<Tensor<TVal, rank, options, Index>, Device>;

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MappedTensor<TensorFixedSize<TVal, Dims, options, Index>, Device>;

// Number of threads to be used for evaluation
inline unsigned int numThreads() {
	# ifdef _OPENMP
		return omp_get_max_threads();
	# else
		return std::thread::hardware_concurrency();
	# endif
}

}

namespace fsc {
	
template<typename T>
Vec3<T> EIGEN_DEVICE_FUNC cross(const Vec3<T>& t1, const Vec3<T>& t2) {	
	Vec3<T> result;
	result(0) = t1(1) * t2(2) - t2(1) * t1(2);
	result(1) = t1(2) * t2(0) - t2(2) * t1(0);
	result(2) = t1(0) * t2(1) - t2(0) * t1(1);
	
	return result;
}

// Inline functions required to instantiate kernel launches
	
#ifdef EIGEN_USE_GPU

namespace internal {
	template<typename Func, typename... Params>
	void __global__ wrappedKernel(Func func, Params... params) {
		func(threadIdx.x, params...);
	}

	template<typename Func, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, Func f, size_t n, Params... params) {
		internal::wrappedKernel<Func, Params...> <<< n, 1, 0, device.stream() >>> (f, params...);
	}
	
	#define INSTANTIATE_KERNEL(func, ...) \
		template void gpuLaunch(Eigen::GpuDevice&, decltype((func)), size_t, __VA_ARGS__);
}

#endif

} // namespace fsc
	