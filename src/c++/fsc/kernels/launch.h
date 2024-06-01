#pragma once

#include <kj/async.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#include "device.h"
#include "../local.h"
#include "../memory.h"

/**
 * \defgroup kernelAPI User API for kernel submission.
 *           Use the methods in here to map data onto the GPU and submit kernels for execution.
 *
 * \defgroup kernelSupport Library support for kernel submission.
 *           Specialize the methods in here to implement custom kernel backends or customized
 *           the mapping of your classes onto GPUs.
 *
 * \defgroup kernels Computation kernels
 */

namespace fsc {
	// Number of threads to be used for evaluation
	inline unsigned int numThreads() {
		# ifdef _OPENMP
			return omp_get_max_threads();
		# else
			return std::thread::hardware_concurrency();
		# endif
	}
	
	
	
	/**
	 * Helper to launch an int-based kernel on a specific device. Currently supports thread-pool- and GPU devices.
	 * \ingroup kernelSupport
	 */
	template<typename Device>
	struct KernelLauncher {
		/** Launches f on the computing capabilities owned by device. The cost parameter should contain an approximation
		 *  of the computing expense for the kernel. On the thread-pool backend, if the expense is sufficiently small,
		 *  the kernel will be computed in-line or with fewer threads.
		 *
		 *  The kernel will run asynchronously. It is guaranteed that it will not start before this function has returned.
		 */
		template<typename Kernel, Kernel f, typename... Params>
		static Promise<void> launch(Device& device, size_t n, Eigen::TensorOpCost& cost, Promise<void> onCancel, Params... params) {
			static_assert(sizeof(Device) == 0, "Kernel launcher not implemented / enabled for this device.");
			return READY_NOW;
		}
	};
	
	// Default builtin kernel launchers
	
	template<>
	struct KernelLauncher<DeviceBase>;
	
	template<>
	struct KernelLauncher<CPUDevice>;
	
	template<>
	struct KernelLauncher<LoopDevice>;
	
	#ifdef FSC_WITH_GPU
	template<>
	struct KernelLauncher<GPUDevice>;
	#endif
	
	namespace internal {					
		/**
		 * Helper function that actually launches the kernel. Needed to perform index expansion
		 * over the parameter. Implemented in launch-inl.h
		 */
		template<typename Kernel, Kernel f, typename Device, typename... Params, size_t... i>
		Promise<void> auxKernelLaunch(Device& device, size_t n, Promise<void> prerequisite, Eigen::TensorOpCost cost, std::index_sequence<i...> indices, Params&&... params);
	}
	
	//! @}
	
	//! \addtogroup kernelAPI
	//! @{
	
	//! Launches the specified kernel on the given device.
	/**
	 *  The exact meaning of the return value depends on the device in question. For CPU devices,
	 *  the fulfillment indicates completion (which is neccessary as memcpy operations
	 *  on this device are immediate). For GPU devices, it only indicates that the
	 *  operation was inserted into the device's compute stream (which is safe as
	 *  memcpy requests are also inserted into the same compute stream) and the preceeding
	 *  memcpy operations all have completed.
	 *
	 *  \note After the returned promise resolves, you have to call fsc::hostMemSynchronize
	 *  on the device and wait on its result before using output parameters.
	 *
	 *  It is recommended to use the macro FSC_LAUNCH_KERNEL(f, ...), which fills in the template
	 *  parameters from its first argument.
	 *
	 *  @code
	 *  auto kernelScheduled = launchKernel<decltype(kernel), kernel>(device, 50, input, output);
	 *  
	 *  // is equivalent to
	 *
	 *  auto kernelScheduled = FSC_LAUNCH_KERNEL(kernel, device, 50, input, output);
	 *  @endcode
	 *
	 *
	 *  \tparam Kernel      Type of the kernel function (usually a decltype expression)
	 *
	 *  \tparam f           Static reference to kernel function
	 *
	 *  \param prerequisite Promise that has to be fulfilled before this kernel will be
	 *                      added to the target device's execution queue.
	 *
	 *  \param device       Eigen Device on which this kernel should be launched.
	 *                      Supported types are:
	 *                       - DefaultDevice
	 *                       - ThreadPoolDevice
	 *                       - GpuDevice (if enabled)
	 *
	 *  \param n            Parameter range for the index (first argument for kernel)
	 *
	 *  \param cost         Cost estimate for the operation. Used by ThreadPoolDevice
	 *                      to determine block size.
	 *
	 *  \param params       Parameters to be mapped to device (if neccessary)
	 *                      and passed to the kernel as additional parameters.
	 *                      For each parameter type P, the kernel will receive
	 *                      a parameter of type DeviceType<P>.
	 *
	 *  \returns A Promise<void> which indicates when additional operations may be submitted
	 *  to the underlying device without interfering with the launched kernel execution
	 *  AND all host -> device copy operations prior to kernel launch are completed (meaning
	 *  that IN-type kernel arguments may be deallocated on the host side).
	 *
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Device& device, const Eigen::TensorOpCost& cost, Promise<void> prerequisite, size_t n, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, mv(prerequisite), cost, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without prerequisite
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Device& device, const Eigen::TensorOpCost& cost, size_t n, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, READY_NOW, cost, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without cost
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Device& device, Promise<void> prerequisite, size_t n, Params&&... params) {
		Eigen::TensorOpCost expensive(1e16, 1e16, 1e16);
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, mv(prerequisite), expensive, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without prerequisite and cost
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Device& device, size_t n, Params&&... params) {
		Eigen::TensorOpCost expensive(1e16, 1e16, 1e16);
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, READY_NOW, expensive, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Launches the given kernel.
	/**
	 * \param f Kernel function
	 * \param ... Parameters to pass to launchKernel()
	 */
	#define FSC_LAUNCH_KERNEL(f, ...) ::fsc::launchKernel<decltype(&f), &f>(__VA_ARGS__)	
	
	//! @}
}

#include "launch-inl.h"