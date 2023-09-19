#pragma once

#include "../eigen.h"

namespace fsc {

// =========================== GPU launcher ==============================

namespace internal {
	#ifdef EIGEN_GPUCC
		template<typename Kernel, Kernel f, typename... Params>
		void __global__ gpuKernel(Params... params) {
			f(blockIdx.x, params...);
		}

		template<typename Kernel, Kernel f, typename... Params>
		void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
			internal::gpuKernel<Kernel, f, Params...> <<< n, 1, 64, device.stream() >>> (params...);
		}
	#else // EIGEN_GPUCC
		template<typename Kernel, Kernel f, typename... Params>
		void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
			static_assert(sizeof(Kernel) == 0, "Please disable instantiation of this kernel .cu files with FSC_DECLARE_KERNEL");
		}
	#endif // EIGEN_GPUCC
}
	
#ifdef FSC_WITH_CUDA
	#ifdef EIGEN_GPUCC

		// Macro to explicitly instantiate kernel
		#define FSC_INSTANTIATE_GPU_KERNEL(func, ...) \
			template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);
		
	#endif // EIGEN_GPUCC

	// Macro to disable implicit instantiation of kernel (which is required to prevent
	#define FSC_DECLARE_KERNEL(func, ...) \
		extern template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);

#else
	#define FSC_DECLARE_KERNEL(func, ...)
#endif // FSC_WITH_CUDA

}