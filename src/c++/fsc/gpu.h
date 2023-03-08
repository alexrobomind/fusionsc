#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define FSC_GPUCC
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	#define FSC_DEVICE_COMPILATION_PHASE
#endif

#if defined(FSC_GPUCC) && !defined(FSC_DEVICE_COMPILATION_PHASE)
	#define FSC_HOST_COMPILATION_PHASE
#endif

#ifdef 

#ifdef __CUDACC__
	#include <cuda.h>
#endif

namespace fsc {
	
#if defined( __HIPCC__)
	using LowLevelGPUStream = hipStream_t;
#elif define(__CUDACC__)
	using LowLevelGPUStream = cudaStream_t;
#endif

struct GPUStream {
	using Callback = kj::Function<void(Maybe<kj::Exception>)>;
	
	GPUStream();
	~GPUStream();
	
	void addCallback(Callback callback);
	void memcpyHostToDevice(kj::byte* dst, const kj::byte* src, size_t s);
	void memcpyDeviceToHost(kj::byte* dst, const kj::byte* src, size_t s);
	
	kj::byte* memAlloc(size_t size);
	void memFree(kj::byte* ptr);
	
	void memRegister(kj::byte* ptr, size_t size);
	void memUnregister(kj::byte* ptr);
	
	kj::byte* translateHostPtr(kj::byte* ptr);
	
	// Required for kernel launches
	LowLevelGPUStream lowLevelStream;
};

}