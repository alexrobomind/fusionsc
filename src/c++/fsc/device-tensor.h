#pragma once

# define EIGEN_USE_THREADS 1
# define EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES 1

#ifdef FSC_WITH_CUDA
	#define EIGEN_USE_GPU
#endif

# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
# include <Eigen/Dense>
# include <cmath>

namespace fsc {
	
using Eigen::Tensor;
using Eigen::TensorFixedSize;
using Eigen::TensorRef;
using Eigen::TensorMap;
using Eigen::Sizes;

constexpr double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp

template<typename T, unsigned int n>
using EVec = Eigen::TensorFixedSize<T, Eigen::Sizes<n>>;

template<typename T>
using EVec3 = EVec<T, 3>;

using EVec3d = EVec3<double>;

/**
 * Helper struct that can be used to map data towards a specific device for calculation. Is constructed with a host pointer and a size,
 * and allocates a corresponding device pointer.
 */
template<typename T, typename Device>
struct MappedData;

/**
 * Tensor constructed on a device sharing a host tensor. Subclasses Eigen::TensorRef<T>.
 */
template<typename T, typename Device>
struct MappedTensor { static_assert(sizeof(T) == 0, "Mapper not implemented"); };

template<typename TVal, int rank, int options, typename Index, typename Device>
struct MappedTensor<Tensor<TVal, rank, options, Index>, Device>;

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MappedTensor<TensorFixedSize<TVal, Dims, options, Index>, Device>;

template<typename... Args>
struct Callback {
	struct BaseHolder {
		virtual void call(Args... args) = 0;
		virtual ~BaseHolder() {}
	};
	
	template<typename T>
	struct Holder : BaseHolder {
		T t;
		Holder(T t) : t(mv(t)) {}
		void call(Args... args) override { t(mv(args)...); }
		~Holder() noexcept {};
	};
	
	BaseHolder* holder;
	
	// Disable copy
	Callback(const Callback<Args...>& other) = delete;
	Callback<Args...>& operator=(const Callback<Args...>& other) = delete;
	Callback(Callback<Args...>&& other) {
		holder = other.holder;
		other.holder = nullptr;
	}
	
	
	template<typename T>
	Callback(T t) :
		holder(new Holder<T>(mv(t)))
	{}
	
	~Callback() { if(holder != nullptr) delete holder; }
	
	void operator()(Args... args) {
		holder->call(args...);
	}
};

}

// Inline implementation

namespace fsc {

namespace internal {

// Inline functions required to instantiate kernel launches

template<typename Device>
void potentiallySynchronize(Device& d) {}

#ifdef EIGEN_USE_GPU

template<>
inline void potentiallySynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& d) {
	d.synchronize();
}

#endif

}

template<typename T, typename Device>
struct MappedData {
	Device& device;
	T* hostPtr;
	T* devicePtr;
	size_t size;
	
	MappedData(Device& device, T* hostPtr, T* devicePtr, size_t size) :
		device(device),
		hostPtr(hostPtr),
		devicePtr(devicePtr),
		size(size)
	{
	}
	
	MappedData(Device& device, T* hostPtr, size_t size) :
		device(device),
		hostPtr(hostPtr),
		devicePtr(deviceAlloc(device, hostPtr, size)),
		size(size)
	{
	}
	
	MappedData(const MappedData& other) = delete;
	MappedData(MappedData&& other) :
		device(other.device),
		hostPtr(other.hostPtr),
		devicePtr(other.devicePtr),
		size(other.size)
	{
		other.devicePtr = nullptr;
	}
	
	~MappedData() {
		if(devicePtr != nullptr) {
			device.deallocate(devicePtr);
		}
	}
	
	void updateHost() {
		device.memcpyDeviceToHost(hostPtr, devicePtr, size * sizeof(T));
	}
	
	void updateDevice() {
		device.memcpyHostToDevice(devicePtr, hostPtr, size * sizeof(T));
	}
	
	static T* deviceAlloc(Device& device, T* hostPtr, size_t size) {
		return (T*) device.allocate(size * sizeof(T));
	}
};

template<typename TVal, int tRank, int tOpts, typename Index, typename Device>
struct MappedTensor<Tensor<TVal, tRank, tOpts, Index>, Device> : public TensorMap<Tensor<TVal, tRank, tOpts, Index>> {
	using Maps = Tensor<TVal, tRank, tOpts, Index>;

	MappedData<TVal, Device> _data;

	MappedTensor(Maps& target, Device& device) :
		TensorMap<Tensor<TVal, tRank, tOpts, Index>>(
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size()),
			target.dimensions()
		),
		_data(device, target.data() /* Host pointer */, this->data() /* Device pointer allocated above */, this->size() /* Same as target.size() */)
	{};

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	TensorMap<Maps> asRef() { return TensorMap<Maps>(*this); }
};

template<typename TVal, typename Dims, int options, typename Index, typename Device>
struct MappedTensor<TensorFixedSize<TVal, Dims, options, Index>, Device> : public TensorMap<TensorFixedSize<TVal, Dims, options, Index>> {
	using Maps = TensorFixedSize<TVal, Dims, options, Index>;
	
	MappedData<TVal, Device> _data;

	MappedTensor(Maps& target, Device& device) :
		TensorMap<TensorFixedSize<TVal, Dims, options, Index>> (
			MappedData<TVal, Device>::deviceAlloc(device, target.data(), target.size())
		),
		_data(target.data(), target.size())
	{}

	void updateHost() { _data.updateHost(); }
	void updateDevice() { _data.updateDevice(); }
	
	TensorRef<Maps> asRef() { return TensorRef<Maps>(*this); }
};

} // namespace fsc
	