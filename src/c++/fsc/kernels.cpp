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