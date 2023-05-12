#include "device.h"
#include "local.h"
#include "eigen.h"

#include <thread>

namespace fsc {
	
// class DeviceBase

int DeviceBase::PTR_SLOT = 0;

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


// === CPUDevice ===

int CPUDevice::BRAND = 0;

CPUDevice::CPUDevice(unsigned int numThreads) :
	DeviceBase(&BRAND),
	eigenDevice(createEigenDevice(numThreads))
{}

CPUDevice::~CPUDevice() {}

Own<Eigen::ThreadPoolDevice> CPUDevice::createEigenDevice(unsigned int numThreads) {
	Own<Eigen::ThreadPoolInterface> threadPool = kj::heap<Eigen::ThreadPool>(numThreads);
	Own<Eigen::ThreadPoolDevice> poolDevice = kj::heap<Eigen::ThreadPoolDevice>(threadPool.get(), numThreads);
	
	return poolDevice.attach(mv(threadPool));
}

void CPUDevice::updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) {
	if(devicePtr == nullptr)
		return;
	
	memcpy(devicePtr, hostPtr, size);
}
	
void CPUDevice::updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) {
	if(devicePtr == nullptr)
		return;
	
	memcpy(hostPtr, devicePtr, size);
}

kj::byte* CPUDevice::map(const kj::byte* hostPtr, size_t size, bool allowAlias) {
	if(allowAlias)
		return nullptr;
	
	return new byte[size];
}

void CPUDevice::unmap(const kj::byte* hostPtr, kj::byte* devicePtr) {
	if(devicePtr == nullptr)
		return;
	
	delete[] devicePtr;
}

kj::byte* CPUDevice::translateToDevice(kj::byte* hostPtr) {
	return hostPtr;
}

Own<DeviceBase> CPUDevice::addRef() {
	return kj::addRef(*this);
}

Promise<void> CPUDevice::emplaceBarrier() {
	return READY_NOW;
}

unsigned int CPUDevice::estimateNumThreads() {
	return std::thread::hardware_concurrency();
}

}