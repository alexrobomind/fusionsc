#pragma once

#include "device-tensor.h"

namespace fsc {
	
#ifdef EIGEN_USE_GPU
#ifdef EIGEN_GPUCC

#pragma message("Providing GPUCC launch infrastructure templates")

namespace internal {
	template<typename Kernel, Kernel f, typename... Params>
	void __global__ wrappedKernel(Params... params) {
		f(blockIdx.x, params...);
	}

	template<typename Kernel, Kernel f, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
		internal::wrappedKernel<Kernel, f, Params...> <<< n, 1, 64, device.stream() >>> (params...);
	}
}

#define INSTANTIATE_KERNEL(func, ...) \
	template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);

#else // EIGEN_GPUCC

namespace internal {
	template<typename Kernel, Kernel f, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
		static_assert(sizeof(Kernel) == 0, "Please disable instantiation of this template outside GPU code with REFERENCE_KERNEL");
	}
}

#pragma message("Referencing GPUCC launch infrastructure templates")
	
#endif // EIGEN_GPUCC


#define REFERENCE_KERNEL(func, ...) \
	extern template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);

#else
	
#define REFERENCE_KERNEL(func, ...)

#pragma message("Ignoring GPUCC launch infrastructure templates")

#endif // EIGEN_USE_GPU
	
}