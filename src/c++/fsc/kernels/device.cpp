#include "device.h"
#include "local.h"
#include "eigen.h"

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
	device -> attach(addRef());
}

void DeviceMappingBase::updateDevice() {
	doUpdateDevice();
	device -> attach(addRef());
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

CPUDevice::CPUDevice(unsigned int numThreads) :
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

// === LoopDevice ===

int LoopDevice::BRAND = 0;

LoopDevice::LoopDevice() :
	CPUDeviceBase(&BRAND)
{}

Own<DeviceBase> LoopDevice::addRef() {
	static LoopDevice instance;
	return kj::attachRef(instance);
}



}