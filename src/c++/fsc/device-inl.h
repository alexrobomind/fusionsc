namespace fsc {

template<typename T>
struct DeviceMapping<kj::Array<T>> : public DeviceMappingBase {	
	kj::Array<T> hostArray;
	kj::byte* devicePtr;
	kj::ArrayPtr<T> deviceArray;
	bool aliased;
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		DeviceMappingBase(device),
		hostArray(mv(array))
	{
		size_t elementCount = hostArray->size();
		kj::byte* devicePtr = device -> map(hostArray->begin(), elementCount * sizeof(T), allowAlias);
		deviceArray = kj::ArrayPtr<T>(devicePtr == nullptr ? device -> translateToDevice(hostArray->begin()) : devicePtr, elementCount * sizeof(T));
	}

	~DeviceMapping() {
		device -> unmap(hostArray -> begin(), devicePtr);
	}

	void doUpdateHost() override {
		device -> updateHost(hostArray -> begin(), devicePtr, hostArray -> size() * sizeof(T));
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(deviceArray.begin(), devicePtr, hostArray -> size() * sizeof(T));
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
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		DeviceMappingBase(device),
		hostArray(mv(array))
	{
		kj::byte* hackedHost = const_cast<kj::byte*>(hostArray->begin());
		size_t elementCount = hostArray->size();
		kj::byte* devicePtr = device -> map(hackedHost, elementCount * sizeof(T), allowAlias);
		deviceArray = kj::ArrayPtr<T>(devicePtr == nullptr ? device -> translateToDevice(hackedHost) : devicePtr, elementCount * sizeof(T));
	}

	~DeviceMapping() {
		device -> unmap(hostArray -> begin(), devicePtr);
	}

	void doUpdateHost() override {
		KJ_FAIL_REQUIRE("Updating const host arrays is illegal");
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(deviceArray.begin(), devicePtr, hostArray -> size() * sizeof(T));
	}
	
	kj::ArrayPtr<T> get() {
		return deviceArray;
	}
};

template<typename T>
struct DeviceMapping<Own<DeviceMapping<T>>> : public DeviceMappingBase {
	Own<DeviceMapping<T>> target;
	
	DeviceMapping(Own<DeviceMapping<T>> otherMapping, DeviceBase& device, bool allowAlias) :
		DeviceMappingBase(device),
		target(mv(otherMapping))
	{}
	
	void doUpdateHost() override { target -> doUpdateHost(); }
	void doUpdateDevice() override { target -> doUpdateDevice(); }
	
	auto get() {
		return target -> get();
	}
};

}