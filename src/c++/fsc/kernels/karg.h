#pragma once

#include "device.h"

namespace fsc {
	
//! Helper type that describes kernel arguments and their in/out semantics.
/**
 * \note For use with FSC_KARG(...)
 */
template<typename T>
struct KernelArg {
	T target;
	
	bool copyToHost = true;
	bool copyToDevice = true;
	bool allowAlias = false;
	
	KernelArg(T in, bool copyToHost, bool copyToDevice, bool allowAlias) :
		target(mv(in)), copyToHost(copyToHost), copyToDevice(copyToDevice)
	{}
};
	
//! Specifies copy behavior for arguments around kernel invocation
enum class KernelArgType {
	NOCOPY = 0,  //!< Don't copy data between host and device for this invocation
	IN = 1,      //!< Only copy data to the device before kernel invocation
	OUT = 2,     //!< Copy data to host after kernel invocation
	INOUT = 3,   //!< Equivalent to IN and OUT
	ALIAS_IN = 5,
	ALIAS_OUT = 6,
	ALIAS_INOUT = 7
};

//! Convenience macro to wrap kernel arguments with the given type
/**
 * \brief Overrides kernel argument type
 * \ingroup kernelAPI
 * 
 * Use this macro to override the
 * transfer behavior of the given argument. 
 *
 * \param val LValue kernel parameter (no temporaries)
 * \param type Kernel parameter type. Possible values for type are
 *         - NOCOPY
 *         - IN  (copy host -> device)
 *         - OUT (cpu device -> host)
 *         - INOUT (both)
 *         - ALIAS_IN (only need copy host -> device, but also allow aliasing)
 *         - ALIAS_OUT (only need copy device -> host, but also allow aliasing)
 *         - ALIAS_INOUT (both copies, allow aliasing)
 */
#define FSC_KARG(val, type) ::fsc::kArg(val, ::fsc::KernelArgType::type)

//! Allows to override the copy behavior on a specified argument
template<typename T>
KernelArg<T> kArg(T in, bool copyToHost, bool copyToDevice, bool allowAlias) { return KernelArg<T>(mv(in), copyToHost, copyToDevice, allowAlias); }

//! Allows to override the copy behavior on a specified argument
template<typename T>
KernelArg<T> kArg(T in, KernelArgType type) {
	return kArg(
		mv(in),
		static_cast<int>(type) & static_cast<int>(KernelArgType::OUT),
		static_cast<int>(type) & static_cast<int>(KernelArgType::IN),
		static_cast<int>(type) & 4
	);
}

//! Allows creation of a kernel arg form reference
template<typename T>
KernelArg<Own<DeviceMapping<T>>> kArg(Own<DeviceMapping<T>>& ref, KernelArgType type) {
	Own<DeviceMappingBase> newRef = ref->addRef();
	return kArg<Own<DeviceMapping<T>>>(newRef.downcast<DeviceMapping<T>>(), type);
}

template<typename T>
struct DeviceMapping<KernelArg<T>> : public DeviceMappingBase {
	DeviceMappingType<T> target;
	bool copyToDevice;
	bool copyToHost;

	DeviceMapping(KernelArg<T>&& arg, DeviceBase& device, bool allowAlias__ignored) :
		DeviceMappingBase(device),
		target(mapToDevice(mv(arg.target), device, arg.allowAlias)),
		copyToDevice(arg.copyToDevice),
		copyToHost(arg.copyToHost)
	{}
	
	void doUpdateHost() override {
		if(copyToHost) target -> doUpdateHost();
	}
	
	void doUpdateDevice() override {
		if(copyToDevice) target -> doUpdateDevice();
	}
	
	auto get() {
		return target -> get();
	}
};

}