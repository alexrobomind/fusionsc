#include "device.h"

namespace fsc {

namespace {

struct CallOnExpire : public kj::AtomicRefcounted {
	using F = kj::Function<void()>;
	F f;
	
	CallOnExpire(F f) : f(mv(f)) {}
	~CallOnExpire() { f(); }
};

}
	
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

void DeviceBase::releaseAttachments() {
	Own<int> result = mv(attachments);
	attachments = kj::Own<int>(&PTR_SLOT, kj::NullDisposer::instance);
	return result;
}

// class DeviceMappingBase

DeviceMappingBase::DeviceMappingBase(DeviceBase& device) :
	device(device.addRef())
{}

DeviceMappingBase::~DeviceMappingBase();

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

CPUDevice::CPUDevice() :
	DeviceBase(&BRAND)
{}

CPUDevice::~CPUDevice() {}

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

kj::byte* translateToDevice(kj::byte* hostPtr) {
	return hostPtr;
}

}