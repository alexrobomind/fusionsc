#pragma once

#include "../common.h"
#include "../eigen.h"

#include <type_traits>

namespace Eigen {
	struct ThreadPoolDevice;
	struct GpuDevice;
}

namespace fsc {

struct DeviceBase {		
	DeviceBase(void* brand);
	virtual ~DeviceBase();
	
	template<typename... T>
	void addToBarrier(T&&... args) {
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
	
	void updateHost();
	void updateDevice();
	Own<DeviceMappingBase> addRef();

protected:
	virtual void doUpdateHost() = 0;
	virtual void doUpdateDevice() = 0;
	mutable Own<DeviceBase> device;
};

namespace internal {
	template<typename T>
	struct IsTriviallyCopyable_ {
		static constexpr bool value = std::is_trivially_copyable<T>::value;
	};
}

#define FSC_DECLARE_TRIVIALLY_COPYABLE(T) \
	namespace internal { \
		template<> \
		struct IsTriviallyCopyable_<T> { \
			static constexpr bool value = true; \
		}; \
	}

template<typename T>
constexpr bool isTriviallyCopyable() {
	return internal::IsTriviallyCopyable_<T>::value;
}

template<typename T>
struct DeviceMapping : public DeviceMappingBase {
	T target;
	
	DeviceMapping(T newTarget, DeviceBase& device, bool allowAlias) :
		DeviceMappingBase(device),
		target(newTarget)
	{
		static_assert(isTriviallyCopyable<T>(), "Default mappings only work for trivially copyable types");
	}

	void doUpdateDevice() override {}
	void doUpdateHost() override {}
	
	T get() { return target; }
};

template<typename T>
Own<DeviceMapping<T>> mapToDevice(T t, DeviceBase& device, bool allowAlias) {
	return kj::refcounted<DeviceMapping<T>>(mv(t), device, allowAlias);
}

template<typename T>
Own<DeviceMapping<T>> mapToDevice(Own<DeviceMapping<T>>&& mapping, DeviceBase& device, bool allowAlias) {
	return mv(mapping);
}

template<typename T>
Own<DeviceMapping<T>> mapToDevice(Own<DeviceMapping<T>>& mapping, DeviceBase& device, bool allowAlias) {
	return mapping -> addRef().template downcast<DeviceMapping<T>>();
}

template<typename T>
using DeviceMappingType = decltype(mapToDevice(std::declval<T>(), std::declval<DeviceBase&>(), true));

template<typename T>
using DeviceType = decltype(std::declval<DeviceMappingType<T>>() -> get());

// ---------------------- Devices ----------------------------------------------------

struct CPUDeviceBase : public DeviceBase {
	using DeviceBase::DeviceBase;
	
	void updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) override;
	void updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) override;
	kj::byte* map(const kj::byte* hostPtr, size_t size, bool allowAlias) override;
	void unmap(const kj::byte* hostPtr, kj::byte* devicePtr) override;
	
	kj::byte* translateToDevice(kj::byte* hostPtr) override;
	
	Promise<void> emplaceBarrier() override;
};

struct CPUDevice : public CPUDeviceBase, public kj::Refcounted {
	static int BRAND;
	
	CPUDevice(kj::Badge<CPUDevice>, unsigned int numThreads);
	~CPUDevice();
	
	static Own<CPUDevice> create(unsigned int numThreads);
	
	Own<DeviceBase> addRef() override;
	
	static unsigned int estimateNumThreads();
	
	// Own<Eigen::ThreadPoolDevice> eigenDevice;
	Eigen::ThreadPoolDevice& eigenDevice();
private:
	unsigned int numThreads;
	Maybe<Own<Eigen::ThreadPoolDevice>> ownDevice;// Own<Eigen::ThreadPoolDevice> eigenDevice;
};

struct LoopDevice : public CPUDeviceBase, public kj::Refcounted {
	static int BRAND;
	
	LoopDevice(kj::Badge<LoopDevice>);
	Own<DeviceBase> addRef() override;
	
	static Own<LoopDevice> create();
};

/*
struct GPUDevice : public DeviceBase, public kj::Refcounted {
};
*/

}