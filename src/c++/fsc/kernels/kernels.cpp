#include "kernels.h"

namespace fsc {

Own<Eigen::ThreadPoolDevice> newThreadPoolDevice() {
	auto pool = kj::heap<Eigen::ThreadPool>(numThreads());
	auto dev  = kj::heap<Eigen::ThreadPoolDevice>(pool.get(), numThreads());
	dev = dev.attach(mv(pool));
	
	return dev;
}

Own<Eigen::DefaultDevice> newDefaultDevice() {
	return kj::heap<Eigen::DefaultDevice>();
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
	// Rescue the op into the stack a.s.a.p.
	Own<const Operation>* typedUserData = (Own<const Operation>*) userData;
	Own<const Operation> op = mv(*typedUserData);
	delete typedUserData; // new in synchronizeGpuDevice
	
	if(status == gpuSuccess) {
		op->done();
		return;
	}
	
	op->fail(KJ_EXCEPTION(FAILED, "GPU computation failed", status));
}

}

/**
 * Schedules a promise to be fulfilled when all previous calls on the GPU device's command stream are finished.
 */
void synchronizeGpuDevice(Eigen::GpuDevice& device, const Operation& op) {	
	// POTENTIALLY UNSAFE
	// Note: We REALLY trust that the callback will always be called, otherwise this is a memory leak
	auto heapOp = new Own<const Operation>(); // delete in gpuSynchCallback
	*heapOp = op.addRef();
	
	# ifdef FSC_WITH_HIP
	auto result = hipStreamAddCallback (device.stream(), gpuSynchCallback, (void*) heapOp, 0);
	# else
	auto result = cudaStreamAddCallback(device.stream(), gpuSynchCallback, (void*) heapOp, 0);
	# endif

	// If the operation failed, we can't trust the callback to be called
	// Better just fail now
	if(result != gpuSuccess) {
		op.fail(KJ_EXCEPTION(FAILED, "Error setting up GPU synchronization callback", result));
		
		// TODO: Verify whether this is really neccessary
		// Potentially the function will actually be never called and the fulfiller can be deleted
		
		// We don't know for sure whether the function will be called or not
		// Better eat a tiny memory leak than calling undefined behavior (so no delete)
		*heapOp = nullptr;
		KJ_FAIL_REQUIRE("Callback scheduling returned error code", result);
	}
}

/*void synchronizeGpuDeviceBlocking(Eigen::GpuDevice& device) {
	device.synchronize();
}*/
	
#endif

}