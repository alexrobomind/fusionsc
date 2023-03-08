namespace fsc {

template<typename T>
struct DeviceMapping<kj::Array<T>> : public DeviceMappingBase {	
	kj::Array<T> hostArray;
	kj::byte* devicePtr;
	kj::ArrayPtr<T> deviceArray;
	bool aliased;
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		MappingBase(device),
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
		MappingBase(device),
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

struct CPUDevice : public DeviceBase, public kj::Refcounted {
	static int BRAND;
	
	CPUDevice();
	~CPUDevice();
	
	void updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) override;
	void updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) override;
	kj::byte* map(const kj::byte* hostPtr, size_t size, bool allowAlias) override;
	void unmap(const kj::byte* hostPtr, kj::byte* devicePtr) override;
	
	kj::byte* translateToDevice(kj::byte* hostPtr) override ;
	
	Own<DeviceBase> addRef() override;
	
	Own<Eigen::ThreadPoolDevice> eigenDevice;
};

struct GPUDevice : public DeviceBase, public kj::Refcounted {
	
};

}