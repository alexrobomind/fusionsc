#pragma once

# define EIGEN_USE_THREADS 1

# define EIGEN_PERMANENTLY_ENABLE_GPU_HIP_CUDA_DEFINES 1

#ifdef CUDA
	#define EIGEN_USE_GPU
#endif

#ifdef HIP
	#define EIGEN_USE_GPU
#endif

# include <unsupported/Eigen/CXX11/Tensor>
# include <unsupported/Eigen/CXX11/ThreadPool>
# include <Eigen/Dense>
# include <cmath>
# include <thread>

# ifdef _OPENMP
	# include <omp.h>
# endif

#define FSC_OFFLOAD 0

namespace fsc {
	using Eigen::Tensor;
	using Eigen::TensorFixedSize;
	using Eigen::TensorRef;
	using Eigen::TensorMap;
	using Eigen::Sizes;
	
	constexpr double pi = 3.14159265358979323846; // "Defined" in magnetics.cpp
	
	template<typename T>
	using Vec3 = TensorFixedSize<T, Sizes<3>>;

	using Vec3d = Vec3<double>;
	using Vec3f = Vec3<float>;

	
	template<typename T>
	typename T::Scalar normSq(const T& t) { TensorFixedSize<typename T::Scalar, Sizes<>> result = t.square().sum(); return result(); }
	
	template<typename T>
	typename T::Scalar norm(const T& t) { return sqrt(normSq(t)); }
	
	template<typename T1, typename T2>
	Vec3<typename T1::Scalar> cross(const T1& t1, const T2& t2);
	
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
	
	// Number of threads to be used for evaluation
	inline unsigned int numThreads() {
		# ifdef _OPENMP
			return omp_get_max_threads();
		# else
			return std::thread::hardware_concurrency();
		# endif
	}
	
	/**
	 * Helper to launch an int-based kernel on a specific device. Currently supports thread-pool- and GPU devices.
	 */
	template<typename Device>
	struct KernelLauncher {
		/** Launches f on the computing capabilities owned by device. The cost parameter should contain an approximation
		 *  of the computing expense for the kernel. On the thread-pool backend, if the expense is sufficiently small,
		 *  the kernel will be computed in-line or with fewer threads.
		 *
		 *  The kernel will run asynchronously. It is guaranteed that it will not start before this function has returned.
		 */
		template<typename Func, typename... Params>
		static Promise<void> launch(Device& device, Func f, size_t n, Eigen::TensorOpCost& cost, Params... params) {
			static_assert(sizeof(Device) == 0, "Kernel launcher not implemented / enabled for this device.");
			return READY_NOW;
		}
	};
	
	/**
	 * Short hand method to launch an expensive kernel. Uses the kernel launcher to launch the
	 * given kernel with a high cost estimate to ensure full parallelization.
	 */
	template<typename Device, typename Func, typename... Params>
	Promise<void> launchExpensiveKernel(Device& device, Func f, size_t n, Params... params) {
		Eigen::TensorOpCost expensive(1e12, 1e12, 1e12);
		return KernelLauncher<Device>::launch(device, mv(f), n, params...);
	}
	
	// === Specializations of kernel launcher ===
	
	template<>
	struct KernelLauncher<Eigen::DefaultDevice>;
	
	template<>
	struct KernelLauncher<Eigen::ThreadPoolDevice>;
	
	#ifdef EIGEN_USE_GPU
	template<>
	struct KernelLauncher<Eigen::GpuDevice>;
	#endif
	
}

// Implementation

namespace fsc {

template<typename T1, typename T2>
Vec3<typename T1::Scalar> cross(const T1& t1, const T2& t2) {
	using Num = typename T1::Scalar;
	
	Vec3<Num> r1 = t1;
	Vec3<Num> r2 = t2;
	
	Vec3<Num> result;
	result(0) = r1(1) * r2(2) - r2(1) * r1(2);
	result(1) = r1(2) * r2(0) - r2(2) * r1(0);
	result(2) = r1(0) * r2(1) - r2(0) * r1(1);
	
	return result;
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
	{}
	
	MappedData(Device& device, T* hostPtr, size_t size) :
		device(device),
		hostPtr(hostPtr),
		devicePtr(deviceAlloc(device, hostPtr, size)),
		size(size)
	{}
	
	~MappedData() {
		device.deallocate(devicePtr);
	}
	
	void updateHost() {
		device.memcpyHostToDevice(hostPtr, devicePtr, size * sizeof(T));
	}
	
	void updateDevice() {
		device.memcpyDeviceToHost(devicePtr, hostPtr, size * sizeof(T));
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


template<>
struct KernelLauncher<Eigen::DefaultDevice> {
	template<typename Func, typename... Params>
	static Promise<void> launch(Eigen::DefaultDevice& device, Func f, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		for(size_t i = 0; i < n; ++i)
			f(i, params...);
		
		return READY_NOW;
	}
};


template<>
struct KernelLauncher<Eigen::ThreadPoolDevice> {
	template<typename Func, typename... Params>
	static Promise<void> launch(Eigen::ThreadPoolDevice& device, Func f, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		auto func = [f = mv(f), params...](size_t i) {
			return f(i, params...);
		};
		
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		auto done = [fulfiller = mv(paf.fulfiller)]() {
			fulfiller->fulfill();
		};
		
		return kj::evalLater([func = mv(func), done = mv(done), cost, n, &device]() {
			device.parallelForAsync(n, cost, func, done);
		});
	}
};

#ifdef EIGEN_USE_GPU

namespace internal {
	
inline void gpuSynchCallback(gpuStream_t stream, gpuError_t status, void* userData) {
	using Fulfiller = CrossThreadPromiseFulfiller<void>;
	
	// Rescue the fulfiller into the stack a.s.a.p.
	Own<Fulfiller>* typedUserData = userData;
	Own<Fulfiller> fulfiller = mv(*typedUserData);
	delete typedUserData; // new in synchronizeGpuDevice
	
	if(fulfiller.get() != nullptr) {
		if(status == gpuSuccess) {
			fulfiller->fulfill();
			return;
		}
		
		fulfiller->reject(KJ_EXCEPTION("GPU computation failed", status));
	}
}

}

/**
 * Schedules a promise to be fulfilled on the GPU device.
 */
Promise<void> synchronizeGpuDevice(Eigen::GpuDevice& device) {
	// Schedule synchronization
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	
	// POTENTIALLY UNSAFE
	// Note: We REALLY trust that the callback will always be called, otherwise this is a memory leak
	auto fulfiller = new Own<CrossThreadPromiseFulfiller<void>>(nullptr); // delete in internal::gpuSynchCallback
	*fulfiller = mv(paf.fulfiller);
	
	try:
	# ifdef CUDA
		auto result = cudaStreamAddCallback(device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# elseif HIP
		auto result = hipStreamAddCallback (device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# endif
	
	// If the operation failed, we can't trust the callback to be called
	// Better just fail now
	if(result != gpuSuccess) {
		*fulfiller = nullptr;
		KJ_FAIL_REQUIRE("Callback scheduling returned error code", result);
	}
	
	return mv(paf.promise);
}

template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Func, typename... Params>
	static Promise<void> launch(ThreadPoolDevice& device, Func f, size_t n, Eigen::TensorOpCost& cost, Param... params) {
		#ifndef EIGEN_GPUCC
			static_assert(sizeof(Func) == 0, "KernelLauncher<Eigen::GpuDevice>::launch must be called from within a GPU compiler (e.g. HIP / Cuda)");
		#else
			#ifdef EIGEN_GPU_COMPILE_PHASE
				static_assert(sizeof(Func) == 0, "KernelLauncher<Eigen::GpuDevice>::launch may not be called from device-side code");
			#else
				GPU_LAUNCH_KERNEL(f, n, 1, 0, device, params...);
				return synchronizeGpuDevice(device);
			#endif
		#endif
	}
};

#endif

}