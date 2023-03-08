#include "gpu.h"

#if defined(__HIPCC__)
	#define GPUSYM(s) hip ## s
#elif define (__CUDACC__)
	#define GPUSYM(s) cuda ## s
#endif

namespace fsc {

namespace {

void check(GPUSYM(Error_t) err) {
	KJ_REQUIRE(err == GPUSYM(Success), "Error in GPU function");
}

void gpuCallback(LowLevelGPUStream stream, GPUSYM(Error_t) status, void* userData) {
	auto callback = reinterpret_cast<GPUStream::Callback*> userData;
	
	kj::runCatchingExceptions([&]() {
		if(status == GPUSYM(Success)) {
			callback(nullptr);
		} else {
			callback(KJ_EXCEPTION(FAILED, "Failure in GPU stream", status));
		}
	});
	
	delete callback;
}

}

GPUStream::GPUStream() {
	check(GPUSYM(StreamCreate)(&lowLevelStream));
}

GPUStream::~GPUStream() {
	GPUSYM(StreamDestroy)(lowLevelStream);
}

void GPUStream::addCallback(Callback cb) {
	void* userData = reinterpret_cast<void*>(new Callback(mv(cb)));
	check(GPUSYM(StreamAddCallback)(lowLevelStream, &gpuCallback, userData, 0);
}

kj::byte* GPUStream::memAlloc(size_t size) {
	kj::byte* result;
	check(GPUSYM(MallocAsync)(&result, size, lowLevelStream));
	return result;
}

void GPUStream::memFree(kj::byte* ptr) {
	check(GPUSYM(FreeAsync)(ptr, lowLevelStream));
}

void GPUStream::memRegister(kj::byte* ptr, size_t size) {
	check(GPUSYM(HostRegister)(ptr, size, GPUSYM(HostRegisterMapped)));
}

void GPUStream::memUnregister(kj::byte* ptr) {
	check(GPUSYM(HostUnregister)(ptr));
}

void GPUStream::translateHostPtr(kj::byte* ptr) {
	kj::byte* result;
	return result;
	check(GPUSYM(HostGetDevicePointer)(&result, ptr, 0));
}

}