#pragma once

#include <kj/async.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#include "tensor.h"
	
#ifdef FSC_WITH_CUDA
	#ifdef EIGEN_GPUCC

		// Macro to explicitly instantiate kernel
		#define INSTANTIATE_KERNEL(func, ...) \
			template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);
		
	#endif // EIGEN_GPUCC

	// Macro to disable implicit instantiation of kernel (which is required to prevent
	#define REFERENCE_KERNEL(func, ...) \
		extern template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);

	#else
		
	#define REFERENCE_KERNEL(func, ...)

#endif // FSC_WITH_CUDA

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
		static Promise<void> launch(Device& device, size_t n, Eigen::TensorOpCost& cost, Params... params) {
			static_assert(sizeof(Device) == 0, "Kernel launcher not implemented / enabled for this device.");
			return READY_NOW;
		}
	};
	
	template<typename Device>
	Promise<void> hostMemSynchronize(Device& device) {
		static_assert(sizeof(Device) == 0, "Memcpy synchronization not implemented for this device.");
		return READY_NOW;
	}
	
	
	/**
	 * Short hand method to launch an expensive kernel. Uses the kernel launcher to launch the
	 * given kernel with a high cost estimate to ensure full parallelization.
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchExpensiveKernel(Device& device, size_t n, Params... params) {
		Eigen::TensorOpCost expensive(1e12, 1e12, 1e12);
		return KernelLauncher<Device>::template launch<Kernel, f, Params...>(device, mv(f), n, expensive, params...);
	}
	
	// === Specializations of kernel launcher ===
	
	template<>
	struct KernelLauncher<Eigen::DefaultDevice>;
	
	template<>
	struct KernelLauncher<Eigen::ThreadPoolDevice>;
	
	#ifdef FSC_WITH_CUDA
	template<>
	struct KernelLauncher<Eigen::GpuDevice>;
	#endif
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	Promise<void> hostMemSynchronize<Eigen::DefaultDevice>(Eigen::DefaultDevice& dev) { return READY_NOW; }
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	Promise<void> hostMemSynchronize<Eigen::ThreadPoolDevice>(Eigen::ThreadPoolDevice& dev) { return READY_NOW; }
	
	#ifdef FSC_WITH_CUDA
	
	Promise<void> synchronizeGpuDevice(Eigen::GpuDevice& device);
	
	/**
	 * GPU devices schedule an asynch memcpy onto their stream. We therefore need to wait until the stream has advanced past it.
	 */
	template<>
	Promise<void> hostMemSynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& device) { return synchronizeGpuDevice(device); }
	
	#endif
}


// Inline Implementation

namespace fsc {
	
#ifdef FSC_WITH_CUDA
#ifdef EIGEN_GPUCC

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

#else // EIGEN_GPUCC

namespace internal {
	template<typename Kernel, Kernel f, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
		static_assert(sizeof(Kernel) == 0, "Please disable instantiation of this template outside GPU code with REFERENCE_KERNEL");
	}
}
	
#endif // EIGEN_GPUCC
#endif // FSC_WITH_CUDA


template<>
struct KernelLauncher<Eigen::DefaultDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(Eigen::DefaultDevice& device, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		for(size_t i = 0; i < n; ++i)
			f(i, params...);
		
		return READY_NOW;
	}
};

template<>
struct KernelLauncher<Eigen::ThreadPoolDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(Eigen::ThreadPoolDevice& device, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		auto func = [params...](Eigen::Index start, Eigen::Index end) mutable {
			for(Eigen::Index i = start; i < end; ++i)
				f(i, params...);
		};
		
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		auto done = [fulfiller = mv(paf.fulfiller)]() mutable {
			fulfiller->fulfill();
		};
		
		auto donePtr = kj::heap<decltype(done)>(mv(done));
		auto funcPtr = kj::heap<decltype(func)>(mv(func));
		
		auto doneCopyable = [ptr = donePtr.get()]() {
			(*ptr)();
		};
		
		auto funcCopyable = [ptr = funcPtr.get()](Eigen::Index start, Eigen::Index end) {
			(*ptr)(start, end);
		};
		
		return kj::evalLater([funcCopyable, doneCopyable, cost, n, &device, promise = mv(paf.promise)]() mutable {
			device.parallelForAsync(n, cost, funcCopyable, doneCopyable);
			return mv(promise);
		}).attach(mv(donePtr), mv(funcPtr));
	}
};

#ifdef FSC_WITH_CUDA

namespace internal {

/**
 * Internal callback to be passed to a synchronization barrier that fulfilles the given promise.
 */
inline void gpuSynchCallback(gpuStream_t stream, gpuError_t status, void* userData) {
	using Fulfiller = kj::CrossThreadPromiseFulfiller<void>;
	
	// Rescue the fulfiller into the stack a.s.a.p.
	Own<Fulfiller>* typedUserData = (Own<Fulfiller>*) userData;
	Own<Fulfiller> fulfiller = mv(*typedUserData);
	delete typedUserData; // new in synchronizeGpuDevice
	
	if(fulfiller.get() != nullptr) {
		if(status == gpuSuccess) {
			fulfiller->fulfill();
			return;
		}
		
		// Note: We take a slightly convoluted path here for the purpose of debugging
		// Debuggers can be set to stop execution on throw
		// If we just assign the exception to the fulfiller, this will only trigger
		// the debugger once it is waited on. This way, the exception gets thrown ASAP,
		// caught by the fulfiller, and then thrown again on wait.
		auto throwException = [&]() {
			KJ_FAIL_REQUIRE("GPU computation failed", status);
		};
		
		fulfiller->rejectIfThrows(throwException);
	}
}

}

/**
 * Schedules a promise to be fulfilled when all previous calls on the GPU device's command stream are finished.
 */
Promise<void> synchronizeGpuDevice(Eigen::GpuDevice& device) {
	// Schedule synchronization
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	
	// POTENTIALLY UNSAFE
	// Note: We REALLY trust that the callback will always be called, otherwise this is a memory leak
	auto fulfiller = new Own<kj::CrossThreadPromiseFulfiller<void>>(); // delete in internal::gpuSynchCallback
	*fulfiller = mv(paf.fulfiller);
	
	# ifdef FSC_WITH_HIP
	auto result = hipStreamAddCallback (device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# else
	auto result = cudaStreamAddCallback(device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# endif

	// If the operation failed, we can't trust the callback to be called
	// Better just fail now
	if(result != gpuSuccess) {
		(*fulfiller)->reject(KJ_EXCEPTION(FAILED, "Error setting up GPU synchronization callback", result));
		
		// TODO: Verify whether this is really neccessary
		// Potentially the function will actually be never called and the fulfiller can be deleted
		
		// We don't know for sure whether the function will be called or not
		// Better eat a tiny memory leak than calling undefined behavior (so no delete)
		*fulfiller = nullptr;
		KJ_FAIL_REQUIRE("Callback scheduling returned error code", result);
	}
	
	return mv(paf.promise);
}

template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Kernel, Kernel func, typename... Params>
	static Promise<void> launch(Eigen::GpuDevice& device, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		return kj::evalNow([&device, n, cost, params...]() {
			KJ_LOG(WARNING, "Launching GPU kernel");
			internal::gpuLaunch<Kernel, func, Params...>(device, n, params...);
			
			auto streamStatus = cudaStreamQuery(device.stream());
			KJ_REQUIRE(streamStatus == cudaSuccess || streamStatus == cudaErrorNotReady, "CUDA launch failed", streamStatus, cudaGetErrorName(streamStatus), cudaGetErrorString(streamStatus));
			
			return synchronizeGpuDevice(device);
		});
}
};

#endif // FSC_WITH_CUDA



}