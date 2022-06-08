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

Own<Eigen::GPUDevice> newGPUDevice() {
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


}