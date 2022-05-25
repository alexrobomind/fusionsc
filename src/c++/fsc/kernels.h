#pragma once

#include <kj/async.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#include "eigen.h"
	
#ifdef FSC_WITH_CUDA
	#ifdef EIGEN_GPUCC

		// Macro to explicitly instantiate kernel
		#define INSTANTIATE_KERNEL(func, ...) \
			template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);
		
	#endif // EIGEN_GPUCC

	// Macro to disable implicit instantiation of kernel (which is required to prevent
	#define REFERENCE_KERNEL(func, ...) \
		extern template void ::fsc::internal::gpuLaunch<decltype(&func), &func, __VA_ARGS__>(Eigen::GpuDevice&, size_t, __VA_ARGS__);

#else
		
	#define REFERENCE_KERNEL(func, ...)

#endif // FSC_WITH_CUDA

namespace fsc {

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
		template<typename Kernel, Kernel f, typename... Params>
		static Promise<void> launch(Device& device, size_t n, Eigen::TensorOpCost& cost, Promise<void> prerequisite, Params... params) {
			static_assert(sizeof(Device) == 0, "Kernel launcher not implemented / enabled for this device.");
			return READY_NOW;
		}
	};
	
	template<typename Device>
	Promise<void> hostMemSynchronize(Device& device) {
		static_assert(sizeof(Device) == 0, "Memcpy synchronization not implemented for this device.");
		return READY_NOW;
	}
	
	
	/**
	 * Short hand method to launch an expensive kernel. Uses the kernel launcher to launch the
	 * given kernel with a high cost estimate to ensure full parallelization.
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchExpensiveKernel(Device& device, size_t n, Params... params) {
		Eigen::TensorOpCost expensive(1e12, 1e12, 1e12);
		return KernelLauncher<Device>::template launch<Kernel, f, Params...>(device, mv(f), n, expensive, params...);
	}

	/**
	 * Class for allocating an array on a device and manage a host and device pointer simultaneously
	 */
	template<typename T, typename Device>
	struct MappedData {
		Device& device;
		T* hostPtr;
		T* devicePtr;
		size_t size;
		
		MappedData(Device& device, T* hostPtr, T* devicePtr, size_t size);
		MappedData(Device& device, T* hostPtr, size_t size);
		MappedData(Device& device);
		
		MappedData(const MappedData& other) = delete;
		MappedData(MappedData&& other);
		
		MappedData& operator=(const MappedData& other) = delete;
		MappedData& operator=(MappedData&& other);
		
		~MappedData();
		
		void updateHost();
		void updateDevice();
		
		static T* deviceAlloc(Device& device, T* hostPtr, size_t size);
	};

	/** std::function is copy-constructableand therefore can only
	 *  be used on copy-constructable lambdas. This is a move-only
	 *  variant of it.
	 */
	 
	/*template<typename... Args>
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
	};*/
	
	/** Helper class template that maps a value to a target device.
	 *
	 *  Specialize this template to override how a specified parameter gets converted
	 *  into a device-located parameter.
	 *
	 *  The default implementation just leaves the transfer up to CUDA / HIP.
	 */
	template<typename T, typename Device, typename SFINAE = void>
	struct MapToDevice {		
		inline MapToDevice(const T in, Device& device) :
			value(in)
		{}
		
		inline T get() { return value; }
		
		inline void updateDevice() {}
		inline void updateHost() {}
		
	private:
		T value;
	};
	
	template<typename T, typename Device>
	using DeviceType = decltype(std::declval<MapToDevice<Decay<T>, Device>>().get());
	
	template<typename Device, typename T>
	MapToDevice<T, Device> mapToDevice(T& in, Device& device) { return MapToDevice<T, Device>(in, device); }
	
	/**
	 * Instantiation of MapToDevice that allows reuse of existing mappings.
	 *
	 * /warning This uses a reference to the original mapping, so its lifetime
	 * is bound by the original mapping. Care must be taken that the original
	 * mapping object is kept alive until the scheduled kernel finishes execution.
	 *
	 */
	template<typename T, typename Device>
	struct MapToDevice<MapToDevice<T, Device>, Device> {
		using Delegate = MapToDevice<T, Device>;
		
		inline MapToDevice(Delegate& delegate, Device& device) :
			delegate(delegate)
		{}
		
		inline MapToDevice(Delegate&& delegate, Device& device) :
			delegate((Delegate&) delegate)
		{}
		
		inline auto get() { return delegate.get(); }
		inline void updateDevice() { delegate.updateDevice(); }
		inline void updateHost() { delegate.updateHost(); }
		
	private:
		Delegate& delegate;
	};
	
	
	/**
	 * Guard check for attempt to use cross-device mapppings.
	 */
	template<typename T, typename Device, typename OtherDevice>
	struct MapToDevice<MapToDevice<T, Device>, OtherDevice> {
		static_assert(
			sizeof(T) == 0, "It is illegal to use data mapped"
			"to separate devices. We only check the static portion (device types),"
			"so ensure carefully that the underlying error is fixed"
		);
	};
	
	/** Helper type that describes kernel arguments and their in/out semantics */
	template<typename T>
	struct KernelArg {
		T& ref;
		
		bool copyToHost = true;
		bool copyToDevice = true;
		
		KernelArg(T& in, bool copyToHost, bool copyToDevice) :
			ref(in), copyToHost(copyToHost), copyToDevice(copyToDevice)
		{}
	};
	
	enum class KernelArgType {
		NOCOPY = 0,
		IN = 1,
		OUT = 2,
		INOUT = 3
	};
	
	template<typename T>
	KernelArg<T> kArg(T& in, bool copyToHost, bool copyToDevice) { return KernelArg<T>(in, copyToHost, copyToDevice); }
	
	template<typename T>
	KernelArg<T> kArg(T& in, KernelArgType type) {
		return kArg(
			in,
			static_cast<int>(type) & static_cast<int>(KernelArgType::OUT),
			static_cast<int>(type) & static_cast<int>(KernelArgType::IN)
		);
	}
	
	#define FSC_KARG(val, type) ::fsc::kArg(val, ::fsc::KernelArgType::type)
	
	template<typename T, typename Device>
	struct MapToDevice<KernelArg<T>, Device>;
	
	namespace internal {
		/**
		 * Helper function that actually launches the kernel. Needed to perform index expansion
		 * over the parameter.
		 */
		template<typename Kernel, Kernel f, typename Device, typename... Params, size_t... i>
		Promise<void> auxKernelLaunch(Device& device, size_t n, Promise<void> prerequisite, Eigen::TensorOpCost cost, std::index_sequence<i...> indices, Params&&... params) {
			// Create mappers for input
			auto mappers = kj::heap<std::tuple<MapToDevice<Decay<Params>, Device>...>>(
				MapToDevice<Decay<Params>, Device>(fwd<Params>(params), device)...
			);
						
			// Update device memory
			
			// Note: This is an extremely convoluted looking way of calling updateDevice on all tuple members
			// C++ 17 has a neater way to do this, but we don't wanna require it just yet
			using givemeatype = int[];
			(void) (givemeatype { 0, (std::get<i>(*mappers).updateDevice(), 0)... });
			
			// Insert a barrier that returns when all pending memcpies are finished
			Promise<void> preSync = hostMemSynchronize(device);
			
			// Call kernel
			auto result = KernelLauncher<Device>::template launch<Kernel, f, DeviceType<Params, Device>...>(device, n, cost, mv(prerequisite), std::get<i>(*mappers).get()...);
			
			// After calling kernel, update host memory where requested (might be async)
			return result.then([mappers = mv(mappers), preSync = mv(preSync), &device]() mutable {
				(void) (givemeatype { 0, (std::get<i>(*mappers).updateHost(), 0)... });
				return mv(preSync);
			});
		}
	}
	
	/**
	 * Launches the specified kernel on the given device.
	 *
	 * Note that you have to call fsc::hostMemSynchronize on the device and wait on its
	 * result before using output parameters.
	 *
	 * It is recommended to use FSC_LAUNCH_KERNEL(f, ...), which eliminates the need to
	 * specify the type of the target function.
	 *
	 * Template parameters:
	 *  Kernel - Type of the kernel function (usually a decltype expression)
	 *  f      - Static reference to kernel function
	 *
	 * Method parameters:
	 *  prerequisite - Promise that has to be fulfilled before this kernel will be
	 *                 added to the target device's execution queue.
	 *  device       - Eigen Device on which this kernel should be launched.
	 *                 Supported types are:
	 *                   - DefaultDevice
	 *                   - ThreadPoolDevice
	 *                   - GpuDevice (if enabled)
	 *  n            - Parameter range for the index (first argument for kernel)
	 *  cost         - Cost estimate for the operation. Used by ThreadPoolDevice
	 *                 to determine block size.
	 *  params       - Parameters to be mapped to device (if neccessary)
	 *                 and passed to the kernel as additional parameters.
	 *
	 * Returns:
	 *  A Promise<void> which indicates when additional operations may be submitted
	 *  to the underlying device without interfering with the launched kernel execution
	 *  AND all host -> device copy operations prior to kernel launch are completed (meaning
	 *  that IN-type kernel arguments may be deallocated on the host side).
	 *
	 *  The exact meaning of this depends on the device in question. For CPU devices,
	 *  the fulfillment indicates completion (which is neccessary as memcpy operations
	 *  on this device are immediate). For GPU devices, it only indicates that the
	 *  operation was inserted into the device's compute stream (which is safe as
	 *  memcpy requests are also inserted into the same compute stream) and the preceeding
	 *  memcpy operations all have completed.
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Promise<void> prerequisite, Device& device, size_t n, const Eigen::TensorOpCost& cost, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, mv(prerequisite), mv(cost), std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Promise<void> launchKernel(Device& device, size_t n, const Eigen::TensorOpCost& cost, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, READY_NOW, mv(cost), std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	#define FSC_LAUNCH_KERNEL(x, ...) ::fsc::launchKernel<decltype(&x), &x>(__VA_ARGS__)
	
	// === Specializations of kernel launcher ===
	
	template<>
	struct KernelLauncher<Eigen::DefaultDevice>;
	
	template<>
	struct KernelLauncher<Eigen::ThreadPoolDevice>;
	
	#ifdef FSC_WITH_CUDA
	template<>
	struct KernelLauncher<Eigen::GpuDevice>;
	#endif
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::DefaultDevice>(Eigen::DefaultDevice& dev) { return READY_NOW; }
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::ThreadPoolDevice>(Eigen::ThreadPoolDevice& dev) { return READY_NOW; }
	
	#ifdef FSC_WITH_CUDA
	
	inline Promise<void> synchronizeGpuDevice(Eigen::GpuDevice& device);
	
	/**
	 * GPU devices schedule an asynch memcpy onto their stream. We therefore need to wait until the stream has advanced past it.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& device) { return synchronizeGpuDevice(device); }
	
	#endif
}


// Inline implementation

namespace fsc {

namespace internal {

// Inline functions required to instantiate kernel launches

template<typename Device>
void potentiallySynchronize(Device& d) {}

#ifdef FSC_WITH_CUDA

template<>
inline void potentiallySynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& d) {
	d.synchronize();
}

#endif

}

}


// Inline Implementation

namespace fsc {
	
#ifdef FSC_WITH_CUDA
#ifdef EIGEN_GPUCC

namespace internal {
	template<typename Kernel, Kernel f, typename... Params>
	void __global__ wrappedKernel(Params... params) {
		f(blockIdx.x, params...);
	}

	template<typename Kernel, Kernel f, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
		internal::wrappedKernel<Kernel, f, Params...> <<< n, 1, 64, device.stream() >>> (params...);
	}
}

#else // EIGEN_GPUCC

namespace internal {
	template<typename Kernel, Kernel f, typename... Params>
	void gpuLaunch(Eigen::GpuDevice& device, size_t n, Params... params) {
		static_assert(sizeof(Kernel) == 0, "Please disable instantiation of this template outside GPU code with REFERENCE_KERNEL");
	}
}
	
#endif // EIGEN_GPUCC
#endif // FSC_WITH_CUDA


template<>
struct KernelLauncher<Eigen::DefaultDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(Eigen::DefaultDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> prerequisite, Params... params) {
		return kj::evalLater([=]() {
			for(size_t i = 0; i < n; ++i)
				f(i, params...);
		});
	}
};

template<>
struct KernelLauncher<Eigen::ThreadPoolDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(Eigen::ThreadPoolDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> prerequisite, Params... params) {
		auto func = [params...](Eigen::Index start, Eigen::Index end) mutable {
			for(Eigen::Index i = start; i < end; ++i)
				f(i, params...);
		};
		
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		auto done = [fulfiller = mv(paf.fulfiller)]() mutable {
			fulfiller->fulfill();
		};
		
		auto donePtr = kj::heap<decltype(done)>(mv(done));
		auto funcPtr = kj::heap<decltype(func)>(mv(func));
		
		auto doneCopyable = [ptr = donePtr.get()]() {
			(*ptr)();
		};
		
		auto funcCopyable = [ptr = funcPtr.get()](Eigen::Index start, Eigen::Index end) {
			(*ptr)(start, end);
		};
		
		return prerequisite.then([funcCopyable, doneCopyable, cost, n, &device, promise = mv(paf.promise)]() mutable {
			device.parallelForAsync(n, cost, funcCopyable, doneCopyable);
			return mv(promise);
		}).attach(mv(donePtr), mv(funcPtr));
	}
};

#ifdef FSC_WITH_CUDA

namespace internal {

/**
 * Internal callback to be passed to a synchronization barrier that fulfilles the given promise.
 */
inline void gpuSynchCallback(gpuStream_t stream, gpuError_t status, void* userData) {
	using Fulfiller = kj::CrossThreadPromiseFulfiller<void>;
	
	// Rescue the fulfiller into the stack a.s.a.p.
	Own<Fulfiller>* typedUserData = (Own<Fulfiller>*) userData;
	Own<Fulfiller> fulfiller = mv(*typedUserData);
	delete typedUserData; // new in synchronizeGpuDevice
	
	if(fulfiller.get() != nullptr) {
		if(status == gpuSuccess) {
			fulfiller->fulfill();
			return;
		}
		
		// Note: We take a slightly convoluted path here for the purpose of debugging
		// Debuggers can be set to stop execution on throw
		// If we just assign the exception to the fulfiller, this will only trigger
		// the debugger once it is waited on. This way, the exception gets thrown ASAP,
		// caught by the fulfiller, and then thrown again on wait.
		auto throwException = [&]() {
			KJ_FAIL_REQUIRE("GPU computation failed", status);
		};
		
		fulfiller->rejectIfThrows(throwException);
	}
}

}

/**
 * Schedules a promise to be fulfilled when all previous calls on the GPU device's command stream are finished.
 */
Promise<void> synchronizeGpuDevice(Eigen::GpuDevice& device) {
	// Schedule synchronization
	auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
	
	// POTENTIALLY UNSAFE
	// Note: We REALLY trust that the callback will always be called, otherwise this is a memory leak
	auto fulfiller = new Own<kj::CrossThreadPromiseFulfiller<void>>(); // delete in internal::gpuSynchCallback
	*fulfiller = mv(paf.fulfiller);
	
	# ifdef FSC_WITH_HIP
	auto result = hipStreamAddCallback (device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# else
	auto result = cudaStreamAddCallback(device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# endif

	// If the operation failed, we can't trust the callback to be called
	// Better just fail now
	if(result != gpuSuccess) {
		(*fulfiller)->reject(KJ_EXCEPTION(FAILED, "Error setting up GPU synchronization callback", result));
		
		// TODO: Verify whether this is really neccessary
		// Potentially the function will actually be never called and the fulfiller can be deleted
		
		// We don't know for sure whether the function will be called or not
		// Better eat a tiny memory leak than calling undefined behavior (so no delete)
		*fulfiller = nullptr;
		KJ_FAIL_REQUIRE("Callback scheduling returned error code", result);
	}
	
	return mv(paf.promise);
}

template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Kernel, Kernel func, typename... Params>
	static Promise<void> launch(Eigen::GpuDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> prerequisite, Params... params) {
		return prerequisite.then([&device, n, cost, params...]() {
			KJ_LOG(WARNING, "Launching GPU kernel");
			internal::gpuLaunch<Kernel, func, Params...>(device, n, params...);
			
			auto streamStatus = cudaStreamQuery(device.stream());
			KJ_REQUIRE(streamStatus == cudaSuccess || streamStatus == cudaErrorNotReady, "CUDA launch failed", streamStatus, cudaGetErrorName(streamStatus), cudaGetErrorString(streamStatus));
		});
	}
};

#endif // FSC_WITH_CUDA

// MappedData
	
template<typename T, typename Device>
MappedData<T, Device>::MappedData(Device& device, T* hostPtr, T* devicePtr, size_t size) :
	device(device),
	hostPtr(hostPtr),
	devicePtr(devicePtr),
	size(size)
{
}

template<typename T, typename Device>
MappedData<T, Device>::MappedData(Device& device, T* hostPtr, size_t size) :
	device(device),
	hostPtr(hostPtr),
	devicePtr(deviceAlloc(device, hostPtr, size)),
	size(size)
{
}

template<typename T, typename Device>
MappedData<T, Device>::MappedData(Device& device) :
	device(device),
	hostPtr(nullptr),
	devicePtr(nullptr),
	size(0)
{}


template<typename T, typename Device>
MappedData<T, Device>::MappedData(MappedData&& other) :
	device(other.device),
	hostPtr(other.hostPtr),
	devicePtr(other.devicePtr),
	size(other.size)
{
	other.devicePtr = nullptr;
}

template<typename T, typename Device>
MappedData<T, Device>& MappedData<T, Device>::operator=(MappedData&& other)
{
	KJ_REQUIRE(&device == &other.device);
	
	hostPtr = other.hostPtr;
	devicePtr = other.devicePtr;
	size = other.size;
	
	other.devicePtr = nullptr;
	
	return *this;
}

template<typename T, typename Device>
MappedData<T, Device>::~MappedData() {
	if(devicePtr != nullptr) {
		device.deallocate(devicePtr);
	}
}

template<typename T, typename Device>
void MappedData<T, Device>::updateHost() {
	device.memcpyDeviceToHost(hostPtr, devicePtr, size * sizeof(T));
}

template<typename T, typename Device>
void MappedData<T, Device>::updateDevice() {
	device.memcpyHostToDevice(devicePtr, hostPtr, size * sizeof(T));
}

template<typename T, typename Device>
T* MappedData<T, Device>::deviceAlloc(Device& device, T* hostPtr, size_t size) {
	return (T*) device.allocate(size * sizeof(T));
}

// Mapper specializations

template<typename T, typename Device>
struct MapToDevice<KernelArg<T>, Device> {
	inline MapToDevice(KernelArg<T>&& in, Device& device) :
		delegate(in.ref, device),
		_updateDevice(in.copyToDevice),
		_updateHost(in.copyToHost)
	{}
	
	inline auto get() { return delegate.get(); }
	inline void updateDevice() { if(_updateDevice) delegate.updateDevice(); }
	inline void updateHost()   { if(_updateHost)   delegate.updateHost();   }
	
private:
	MapToDevice<T, Device> delegate;
	bool _updateDevice;
	bool _updateHost;
};

}