#incldue "common.h"

namespace fsc {

struct DeviceBase {	
	inline static int PTR_SLOT = 0;
	
	DeviceBase(void* brand) :
		brand(brand)
	{}
	
	template<typename... T>
	void attach(T&&... args) {
		attachments = attachments.attach(kj::fwd<T>(args)...);
	}
	
	inline void flush() {
		auto barrier = [a = mv(attachments)]() {
			a = nullptr;
		};
		
		resetAttachments();
		emplaceBarrier(mv(barrier));
	}
	
	inline Promise<void> barrier() {
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		
		auto barrier = [a = mv(attachments), f = mv(paf.fulfiller)]() {
			a = nullptr;
			f -> fulfill();
		};
		
		resetAttachments();
		emplaceBarrier(mv(barrier));
		return mv(paf.promise);
	}	
	
	virtual void emplaceBarrier(kj::Function<void()> f) = 0;
	void updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) = 0;
	void updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) = 0;
	
	kj::byte* map(const kj::byte* hostPtr, size_t size, bool allowAlias) = 0;
	void unmap(const kj::byte* hostPtr, kj::byte* devicePtr) = 0;
	
	Own<DeviceBase> addRef() = 0;
	
	const void* brand;
	
private:
	void resetAttachments() {
		attachments = kj::Own<int>(&PTR_SLOT, kj::NullDisposer::instance);
	}
	
	Own<int> attachments;
};

struct CPUDevice : public DeviceBase {
	kj::Vector<Promise<void>> activeTasks;
	
	static inline int DEVICE_BRAND = 0;
	
	CPUDevice() :
		DeviceBase(&DEVICE_BRAND)
	{}
	
	void updateDevice(kj::byte* devicePtr, const kj::byte* hostPtr, size_t size) override {
		if(devicePtr == hostPtr)
			return;
		
		memcpy(devicePtr, hostPtr, size);
	}
	
	void updateHost(kj::byte* hostPtr, const kj::byte* devicePtr, size_t size) override {
		if(devicePtr == hostPtr)
			return;
		
		memcpy(hostPtr, devicePtr, size);
	}
	
	kj::byte* map(const kj::byte* hostPtr, size_t size, bool allowAlias) override {
		if(allowAlias)
			return hostPtr;
		
		return new byte[size];
	}
	
	void unmap(const kj::byte* hostPtr, kj::byte* devicePtr) override {
		if(hostPtr == devicePtr)
			return;
		
		delete[] devicePtr;
	}
};

struct DeviceMappingBase : public kj::AtomicRefcounted {
	DeviceMappingBase(DeviceBase& device) :
		device(device.addRef())
	{}
	
	virtual ~DeviceMappingBase() = 0;
	
	inline void updateHost() {
		doUpdateHost();
		device -> attach(addRef());
	}
	
	inline void updateDevice() {
		doUpdateDevice();
		device -> attach(addRef());
	}
	
	inline Own<const DeviceMappingBase> addRef() const {
		return kj::atomicAddRef(*this);
	}

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
	return kj::atomicRefcounted<DeviceMapping<T>>(t, device, kj::fwd<Args>(args)...)
}

// ---------------------- Mapping type for simple arrays -----------------------------

template<typename T>
struct DeviceMapping<kj::Array<T>> : public DeviceMappingBase {	
	kj::Array<T> hostArray;
	kj::ArrayPtr<T> deviceArray;
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		MappingBase(device),
		hostArray(mv(array))
	{
		size_t elementCount = hostArray->size();
		kj::byte* devicePtr = device -> map(hostArray->begin(), elementCount * sizeof(T), allowAlias);
		deviceArray = kj::ArrayPtr<T>(devicePtr, elementCount * sizeof(T));
	}

	~DeviceMapping() {
		device -> unmap(hostArray -> begin(), deviceArray.begin());
	}

	void doUpdateHost() override {
		device -> updateHost(hostArray -> begin(), deviceArray.begin(), hostArray -> size() * sizeof(T));
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(deviceArray.begin(), hostArray -> begin(), hostArray -> size() * sizeof(T));
	}
	
	kj::ArrayPtr<T> get() {
		return deviceArray;
	}
};

template<typename T>
struct DeviceMapping<kj::Array<const T>> : public DeviceMappingBase {	
	kj::Array<const T> hostArray;
	kj::ArrayPtr<T> deviceArray;
	
	DeviceMapping(kj::Array<T> array, DeviceBase& device, bool allowAlias = false) :
		MappingBase(device),
		hostArray(mv(array))
	{
		size_t elementCount = hostArray->size();
		kj::byte* devicePtr = device -> map(hostArray->begin(), elementCount * sizeof(T), allowAlias);
		deviceArray = kj::ArrayPtr<T>(devicePtr, elementCount * sizeof(T));
	}

	~DeviceMapping() {
		device -> unmap(hostArray -> begin(), deviceArray.begin());
	}

	void doUpdateHost() override {
		KJ_FAIL_REQUIRE("Updating const host arrays is illegal");
	}
	
	void doUpdateDevice() override {
		device -> updateDevice(deviceArray.begin(), hostArray -> begin(), hostArray -> size() * sizeof(T));
	}
	
	kj::ArrayPtr<T> get() {
		return deviceArray;
	}
};

}