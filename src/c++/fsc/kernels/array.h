#pragma once

#include "../common.h"
#include "device.h"

namespace fsc {

template<typename T>
struct DeviceMapping<kj::Array<T>> : public DeviceMappingBase {	
	kj::Array<T> hostArray;
	kj::byte* devicePtr;
	kj::ArrayPtr<T> deviceArray;
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		DeviceMappingBase(device),
		hostArray(mv(array))
	{
		size_t elementCount = hostArray.size();
		kj::byte* hostPtr = reinterpret_cast<kj::byte*>(hostArray.begin());
		devicePtr = device.map(hostPtr, elementCount * sizeof(T), allowAlias);
		kj::byte* actualDevicePtr = devicePtr == nullptr ? device.translateToDevice(hostPtr) : devicePtr;
		deviceArray = kj::ArrayPtr<T>(reinterpret_cast<T*>(actualDevicePtr), elementCount);
	}

	~DeviceMapping() {
		device -> unmap(reinterpret_cast<kj::byte*>(hostArray.begin()), devicePtr);
	}

	void doUpdateHost() override {
		device -> updateHost(reinterpret_cast<kj::byte*>(hostArray.begin()), devicePtr, hostArray.size() * sizeof(T));
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(devicePtr, reinterpret_cast<kj::byte*>(hostArray.begin()), hostArray.size() * sizeof(T));
	}
	
	kj::ArrayPtr<T> get() {
		return deviceArray;
	}
};

template<typename T>
struct DeviceMapping<kj::Array<const T>> : public DeviceMappingBase {	
	kj::Array<const T> hostArray;
	kj::byte* devicePtr;
	kj::ArrayPtr<T> deviceArray;
	
	DeviceMapping(kj::Array<const T> array, DeviceBase& device, bool allowAlias = false) :
		DeviceMappingBase(device),
		hostArray(mv(array))
	{
		kj::byte* hackedHost = const_cast<kj::byte*>(reinterpret_cast<const kj::byte*>(hostArray.begin()));
		size_t elementCount = hostArray.size();
		devicePtr = device.map(hackedHost, elementCount * sizeof(T), allowAlias);
		kj::byte* actualDevicePtr = devicePtr == nullptr ? device.translateToDevice(hackedHost) : devicePtr;
		deviceArray = kj::ArrayPtr<T>(reinterpret_cast<T*>(actualDevicePtr), elementCount);
	}

	~DeviceMapping() {
		device -> unmap(const_cast<kj::byte*>(reinterpret_cast<const kj::byte*>(hostArray.begin())), devicePtr);
	}

	void doUpdateHost() override {
		KJ_FAIL_REQUIRE("Updating const host arrays is illegal");
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(devicePtr, reinterpret_cast<const kj::byte*>(deviceArray.begin()), hostArray.size() * sizeof(T));
	}
	
	kj::ArrayPtr<T> get() {
		return deviceArray;
	}
};

}