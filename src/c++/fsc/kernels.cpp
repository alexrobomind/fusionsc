#include "kernels.h"

namespace fsc {

Own<Eigen::ThreadPoolDevice> newThreadPoolDevice() {
	auto pool = kj::heap<Eigen::ThreadPool>(numThreads());
	auto dev  = kj::heap<Eigen::ThreadPoolDevice>(pool.get(), numThreads());
	dev = dev.attach(mv(pool));
	
	return dev;
}


#ifdef FSC_WITH_CUDA

#include <cuda_runtime_api.h>

Own<Eigen::GpuDevice> newGpuDevice() {
	auto stream = kj::heap<Eigen::GpuStreamDevice>();
	
	cudaError_t streamStatus = cudaStreamQuery(stream->stream());
	KJ_REQUIRE(streamStatus == cudaSuccess, "CUDA stream could not be initialized", streamStatus);
	
	auto dev    = kj::heap<Eigen::GpuDevice>(stream);
	
	streamStatus = cudaStreamQuery(stream->stream());
	KJ_REQUIRE(streamStatus == cudaSuccess, "CUDA device could not be initialized", streamStatus);
	
	dev = dev.attach(mv(stream));
	
	return dev;
}

namespace {

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
	auto fulfiller = new Own<kj::CrossThreadPromiseFulfiller<void>>(); // delete in gpuSynchCallback
	*fulfiller = mv(paf.fulfiller);
	
	# ifdef FSC_WITH_HIP
	auto result = hipStreamAddCallback (device.stream(), gpuSynchCallback, (void*) fulfiller, 0);
	# else
	auto result = cudaStreamAddCallback(device.stream(), gpuSynchCallback, (void*) fulfiller, 0);
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

void synchronizeGpuDeviceBlocking(Eigen::GpuDevice& device) {
	device.synchronize();
}
	
#endif

namespace {
	
struct ErrorHandler : public kj::TaskSet::ErrorHandler {
	void taskFailed(kj::Exception&& exception) override {
		KJ_LOG(WARNING, "Update Host task failed", exception);
	}
};

ErrorHandler errorHandler;

}

thread_local kj::TaskSet internal::kernelDaemon(errorHandler);

}