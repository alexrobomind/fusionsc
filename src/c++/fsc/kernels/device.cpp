#include "device.h"
#include "../local.h"
#include "../eigen.h"

#include <thread>

namespace fsc {
	
// class DeviceBase

namespace {
	int PTR_SLOT = 0;
}

DeviceBase::DeviceBase(void* brand) :
	brand(brand)
{
	releaseAttachments();
}

DeviceBase::~DeviceBase() {}

Promise<void> DeviceBase::barrier() {
	return getActiveThread().uncancelable(emplaceBarrier().attach(releaseAttachments(), addRef()));
}

Own<int> DeviceBase::releaseAttachments() {
	Own<int> result = mv(attachments);
	attachments = kj::Own<int>(&PTR_SLOT, kj::NullDisposer::instance);
	return result;
}

// class DeviceMappingBase

DeviceMappingBase::DeviceMappingBase(DeviceBase& device) :
	device(device.addRef())
{}

DeviceMappingBase::~DeviceMappingBase() {}

void DeviceMappingBase::updateHost() {
	doUpdateHost();
	device -> addToBarrier(addRef());
}

void DeviceMappingBase::updateDevice() {
	doUpdateDevice();
	device -> addToBarrier(addRef());
}

Own<DeviceMappingBase> DeviceMappingBase::addRef() {
	return kj::addRef(*this);
}

// === CPUDeviceBase ===

void CPUDeviceBase::updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) {
	if(devicePtr == nullptr)
		return;
	
	memcpy(devicePtr, hostPtr, size);
}
	
void CPUDeviceBase::updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) {
	if(devicePtr == nullptr)
		return;
	
	memcpy(hostPtr, devicePtr, size);
}

kj::byte* CPUDeviceBase::map(const kj::byte* hostPtr, size_t size, bool allowAlias) {
	if(allowAlias)
		return nullptr;
	
	return new byte[size];
}

void CPUDeviceBase::unmap(const kj::byte* hostPtr, kj::byte* devicePtr) {
	if(devicePtr == nullptr)
		return;
	
	delete[] devicePtr;
}

kj::byte* CPUDeviceBase::translateToDevice(kj::byte* hostPtr) {
	return hostPtr;
}

Promise<void> CPUDeviceBase::emplaceBarrier() {
	return READY_NOW;
}

// === CPUDevice ===

namespace { 
	Own<Eigen::ThreadPoolDevice> createEigenDevice(unsigned int numThreads) {
		Own<Eigen::ThreadPoolInterface> threadPool = kj::heap<Eigen::ThreadPool>(numThreads);
		Own<Eigen::ThreadPoolDevice> poolDevice = kj::heap<Eigen::ThreadPoolDevice>(threadPool.get(), numThreads);
		
		return poolDevice.attach(mv(threadPool));
	}
}

int CPUDevice::BRAND = 0;

CPUDevice::CPUDevice(kj::Badge<CPUDevice>, unsigned int numThreads) :
	CPUDeviceBase(&BRAND),
	eigenDevice(createEigenDevice(numThreads))
{}

CPUDevice::~CPUDevice() {}

Own<DeviceBase> CPUDevice::addRef() {
	return kj::addRef(*this);
}

unsigned int CPUDevice::estimateNumThreads() {
	return std::thread::hardware_concurrency();
}

Own<CPUDevice> CPUDevice::create(unsigned int numThreads) {
	return kj::refcounted<CPUDevice>(kj::Badge<CPUDevice>(), numThreads);
}

// === LoopDevice ===

int LoopDevice::BRAND = 0;

LoopDevice::LoopDevice(kj::Badge<LoopDevice>) :
	CPUDeviceBase(&BRAND)
{}

Own<DeviceBase> LoopDevice::addRef() {
	return kj::addRef(*this);
}

Own<LoopDevice> LoopDevice::create() {
	return kj::refcounted<LoopDevice>(kj::Badge<LoopDevice>());
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

#endif

}