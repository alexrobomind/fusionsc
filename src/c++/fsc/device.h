#pragma once

#include "common.h"

namespace Eigen {
	struct ThreadPoolDevice;
	struct GpuDevice;
}

namespace fsc {

struct DeviceBase {	
	static int PTR_SLOT;
	
	DeviceBase(void* brand);
	virtual ~DeviceBase();
	
	template<typename... T>
	void attach(T&&... args) {
		attachments = attachments.attach(kj::fwd<T>(args)...);
	}
	
	Promise<void> barrier();
	
	virtual void updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) = 0;
	virtual void updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) = 0;
	
	virtual kj::byte* map(const kj::byte* hostPtr, size_t size, bool allowAlias) = 0;
	virtual void unmap(const kj::byte* hostPtr, kj::byte* devicePtr) = 0;
	virtual kj::byte* translateToDevice(kj::byte* hostPtr) = 0;
		
	virtual Own<DeviceBase> addRef() = 0;
	
	const void* brand;

protected:
	virtual Promise<void> emplaceBarrier() = 0;
	Own<int> releaseAttachments();
	
private:
	Own<int> attachments;
};

struct DeviceMappingBase : public kj::Refcounted {
	DeviceMappingBase(DeviceBase& device);
	virtual ~DeviceMappingBase();
	
	inline void updateHost();
	inline void updateDevice();
	inline Own<DeviceMappingBase> addRef();

protected:
	virtual void doUpdateHost() = 0;
	virtual void doUpdateDevice() = 0;
	mutable Own<DeviceBase> device;
};

template<typename T>
struct DeviceMapping {
	static_assert(sizeof(T) == 0, "No device mapping defined for this type");
};

template<typename T>
using DeviceType<T> = decltype(std::declval<DeviceMapping<T>>().get());

template<typename T, typename... Args>
Own<DeviceMapping<T>> mapToDevice(T t, DeviceBase& device, Args... args) {
	return kj::refcounted<DeviceMapping<T>>(t, device, kj::fwd<Args>(args)...)
}

// ---------------------- Mapping type for simple arrays -----------------------------

template<typename T>
struct DeviceMapping<kj::Array<T>>;

template<typename T>
struct DeviceMapping<kj::Array<const T>>;

template<typename T>
struct DeviceMapping<Own<DeviceMapping<T>>;

// ---------------------- Devices ----------------------------------------------------

struct CPUDevice;
struct GPUDevice;

}

#include "device-inl.h"