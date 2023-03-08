#pragma once

#include <kj/async.h>

#ifdef _OPENMP
	#include <omp.h>
#endif

#include "eigen.h"
#include "operation.h"
	
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

/**
 * \defgroup kernelAPI User API for kernel submission.
 *           Use the methods in here to map data onto the GPU and submit kernels for execution.
 *
 * \defgroup kernelSupport Library support for kernel submission.
 *           Specialize the methods in here to implement custom kernel backends or customized
 *           the mapping of your classes onto GPUs.
 *
 * \defgroup kernels Computation kernels
 */

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
	 * \ingroup kernelSupport
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
			return newOperation();
		}
	};
	
	template<>
	struct KernelLauncher<DeviceBase>;
	
	template<>
	struct KernelLauncher<CpuDevice>;
	
	#ifdef FSC_WITH_GPU
	template<>
	struct KernelLauncher<GpuDevice>;
	#endif
	
	//! Helper type that describes kernel arguments and their in/out semantics.
	/**
	 * \note For use with FSC_KARG(...)
	 */
	template<typename T>
	struct KernelArg {
		T& ref;
		
		bool copyToHost = true;
		bool copyToDevice = true;
		bool allowAlias = false;
		
		KernelArg(T& in, bool copyToHost, bool copyToDevice, bool allowAlias) :
			ref(in), copyToHost(copyToHost), copyToDevice(copyToDevice)
		{}
	};
	
	template<typename T>
	struct DeviceMapping<T>;
		
	//! Specifies copy behavior for arguments around kernel invocation
	enum class KernelArgType {
		NOCOPY = 0,  //!< Don't copy data between host and device for this invocation
		IN = 1,      //!< Only copy data to the device before kernel invocation
		OUT = 2,     //!< Copy data to host after kernel invocation
		INOUT = 3,   //!< Equivalent to IN and OUT
		ALIAS_IN = 5,
		ALIAS_INOUT = 7
	};
	
	//! Allows to override the copy behavior on a specified argument
	template<typename T>
	KernelArg<T> kArg(T& in, bool copyToHost, bool copyToDevice) { return KernelArg<T>(in, copyToHost, copyToDevice); }
	
	//! Allows to override the copy behavior on a specified argument
	template<typename T>
	KernelArg<T> kArg(T& in, KernelArgType type) {
		return kArg(
			in,
			static_cast<int>(type) & static_cast<int>(KernelArgType::OUT),
			static_cast<int>(type) & static_cast<int>(KernelArgType::IN),
			static_cast<int>(type) & 4
		);
	}
	
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
	 *         - OUT (cpüy device -> host)
	 *         - INOUT (both)
	 */
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
			auto result = ownHeld(newOperation());
			
			// Create mappers for input
			auto mappers = heapHeld<std::tuple<Own<DeviceMapping<Decay<Params>>>...>>(
				mapToDevice<Decay<Params>>(fwd<Params>(params), device)
			);
						
			using givemeatype = int[];
			
			Promise<void> task = prerequisite.then([&device, mappers]() mutable {
				// Update device memory
				// Note: This is an extremely convoluted looking way of calling updateDevice on all tuple members
				// C++ 17 has a neater way to do this, but we don't wanna require it just yet
				(void) (givemeatype { 0, (std::get<i>(*mappers).updateDevice(), 0)... });
				
				return device.barrier();
			})
			.then([&device, cost, n, mappers]() mutable {
				// Call kernel
				auto promise = KernelLauncher<Device>::template launch<Kernel, f, DeviceType<Params, Device>...>(device, n, cost, std::get<i>(*mappers).get()...);
				promise = promise.attach(device.addRef(), std.get<i>(*mappers).addRef()...);
				return getActiveThread().uncancelable(mv(promise));
			}).then([&device, mappers...]() mutable {
				// Update host memory
				(void) (givemeatype { 0, (std::get<i>(*mappers).updateHost(), 0)... });
				
				return device.barrier();
			})
			.attach(mappers.x(), device.addRef());
			
			return result;
		}
	}
	
	//! @}
	
	//! \addtogroup kernelAPI
	//! @{
	
	//! Launches the specified kernel on the given device.
	/**
	 *  The exact meaning of the return value depends on the device in question. For CPU devices,
	 *  the fulfillment indicates completion (which is neccessary as memcpy operations
	 *  on this device are immediate). For GPU devices, it only indicates that the
	 *  operation was inserted into the device's compute stream (which is safe as
	 *  memcpy requests are also inserted into the same compute stream) and the preceeding
	 *  memcpy operations all have completed.
	 *
	 *  \note After the returned promise resolves, you have to call fsc::hostMemSynchronize
	 *  on the device and wait on its result before using output parameters.
	 *
	 *  It is recommended to use the macro FSC_LAUNCH_KERNEL(f, ...), which fills in the template
	 *  parameters from its first argument.
	 *
	 *  @code
	 *  auto kernelScheduled = launchKernel<decltype(kernel), kernel>(device, 50, input, output);
	 *  
	 *  // is equivalent to
	 *
	 *  auto kernelScheduled = FSC_LAUNCH_KERNEL(kernel, device, 50, input, output);
	 *  @endcode
	 *
	 *
	 *  \tparam Kernel      Type of the kernel function (usually a decltype expression)
	 *
	 *  \tparam f           Static reference to kernel function
	 *
	 *  \param prerequisite Promise that has to be fulfilled before this kernel will be
	 *                      added to the target device's execution queue.
	 *
	 *  \param device       Eigen Device on which this kernel should be launched.
	 *                      Supported types are:
	 *                       - DefaultDevice
	 *                       - ThreadPoolDevice
	 *                       - GpuDevice (if enabled)
	 *
	 *  \param n            Parameter range for the index (first argument for kernel)
	 *
	 *  \param cost         Cost estimate for the operation. Used by ThreadPoolDevice
	 *                      to determine block size.
	 *
	 *  \param params       Parameters to be mapped to device (if neccessary)
	 *                      and passed to the kernel as additional parameters.
	 *                      For each parameter type P, the kernel will receive
	 *                      a parameter of type DeviceType<P>.
	 *
	 *  \returns A Promise<void> which indicates when additional operations may be submitted
	 *  to the underlying device without interfering with the launched kernel execution
	 *  AND all host -> device copy operations prior to kernel launch are completed (meaning
	 *  that IN-type kernel arguments may be deallocated on the host side).
	 *
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Own<Operation> launchKernel(Device& device, const Eigen::TensorOpCost& cost, Promise<void> prerequisite, size_t n, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, mv(prerequisite), cost, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without prerequisite
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Own<Operation> launchKernel(Device& device, const Eigen::TensorOpCost& cost, size_t n, Params&&... params) {
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, READY_NOW, cost, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without cost
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Own<Operation> launchKernel(Device& device, Promise<void> prerequisite, size_t n, Params&&... params) {
		Eigen::TensorOpCost expensive(1e16, 1e16, 1e16);
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, mv(prerequisite), expensive, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Version of launchKernel() without prerequisite and cost
	template<typename Kernel, Kernel f, typename Device, typename... Params>
	Own<Operation> launchKernel(Device& device, size_t n, Params&&... params) {
		Eigen::TensorOpCost expensive(1e16, 1e16, 1e16);
		return internal::auxKernelLaunch<Kernel, f, Device, Params...>(device, n, READY_NOW, expensive, std::make_index_sequence<sizeof...(params)>(), fwd<Params>(params)...);
	}
	
	//! Launches the given kernel.
	/**
	 * \param f Kernel function
	 * \param ... Parameters to pass to launchKernel()
	 */
	#define FSC_LAUNCH_KERNEL(f, ...) ::fsc::launchKernel<decltype(&f), &f>(__VA_ARGS__)
	
	//! Creates a thread pool device
	Own<Eigen::ThreadPoolDevice> newThreadPoolDevice();
	
	//! Creates a thread pool device
	Own<Eigen::DefaultDevice> newDefaultDevice();
	
	#ifdef FSC_WITH_CUDA
	
	//! Creates a GPU device
	Own<Eigen::GpuDevice> newGpuDevice();
	
	#endif
	
	
	//! @}
	
	//! \addtogroup kernelSupport
	//! @{
	
	// === Specializations of kernel launcher ===
	
	#ifdef FSC_WITH_CUDA
	template<>
	struct KernelLauncher<Eigen::GpuDevice>;
	#endif
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::DefaultDevice>(Eigen::DefaultDevice& dev, const Operation& op) { return READY_NOW; }
	/*template<>
	inline void hostMemSynchronizeBlocking<Eigen::DefaultDevice>(Eigen::DefaultDevice& dev) {}*/
	
	/**
	 * CPU devices dont need host memory synchronization, as all memcpy() calls are executed in-line on the main thread.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::ThreadPoolDevice>(Eigen::ThreadPoolDevice& dev, const Operation& op) { return READY_NOW; }
	/*template<>
	inline void hostMemSynchronizeBlocking<Eigen::ThreadPoolDevice>(Eigen::ThreadPoolDevice& dev) {}*/
	
	#ifdef FSC_WITH_CUDA
	
	void synchronizeGpuDevice(Eigen::GpuDevice& device, const Operation& op);
	//void synchronizeGpuDeviceBlocking(Eigen::GpuDevice& device);
	
	/**
	 * GPU devices schedule an asynch memcpy onto their stream. We therefore need to wait until the stream has advanced past it.
	 */
	template<>
	inline Promise<void> hostMemSynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& device, const Operation& op) { auto child = op.newChild(); synchronizeGpuDevice(device, *child); return child -> whenDone(); }
	/*template<>
	inline void hostMemSynchronizeBlocking<Eigen::GpuDevice>(Eigen::GpuDevice& device) { synchronizeGpuDeviceBlocking(device); }*/
	
	#endif
	
	//! @}
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

// Multi-target launcher
template<>
struct KernelLauncher<DeviceBase> {
	template<typename Kernel, Kernel f, typename... Params>
	Promise<void> launch(DeviceBase& device, size_t n, const Eigen::TensorOpCost& cost, Params... params) {
		#define FSC_HANDLE_TYPE(DevType) \
			if(device.brand == DevType::BRAND) \
				return KernelLauncher<DevType>::launch(static_cast<DevType&>(device), n, cost, fwd<Params>(params)...);
				
		FSC_HANDLE_TYPE(CpuDevice);
		
		#ifdef FSC_WITH_CUDA
		FSC_HANDLE_TYPE(GpuDevice);
		#endif
		
		#undef FSC_HANDLE_TYPE
		
		KJ_FAIL_REQUIRE(
			"Unknown device brand. To launch kernels from a DeviceBase reference,"
			" the device must be of one of the following types: fsc::CpuDevice"
			" or fsc::GpuDevice"
		);
	}
};
	
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
	static Own<Operation> launch(Eigen::DefaultDevice& device, size_t n, const Eigen::TensorOpCost& cost, Params... params) {
		auto result = newOperation();
		
		auto task = kj::evalLater([=, result = result->addRef()]() {
			for(size_t i = 0; i < n; ++i)
				f(i, params...);
			result->done();
		});
		result -> dependsOn(mv(task));
		
		return result;
	}
};

template<>
struct KernelLauncher<Eigen::ThreadPoolDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(Eigen::ThreadPoolDevice& device, size_t n, const Eigen::TensorOpCost& cost, Params... params) {
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		AtomicShared<kj::CrossThreadPromiseFulfiller<void>> fulfiller = mv(paf.fulfiller);
		
		struct Context {
			kj::MutexGuarded<Maybe<kj::Exception>> exception;
		};
		AtomicShared<Context> ctx = kj::heap<Context>();
		
		auto func = [ctx, params...](Eigen::Index start, Eigen::Index end) mutable {
			auto maybeException = kj::runCatchingExceptions([params..., start, end]() {
				for(Eigen::Index i = start; i < end; ++i)
					f(i, params...);
			});
			
			// If we failed, transfer exception
			KJ_IF_MAYBE(pErr, maybeException) {
				auto locked = ctx -> exception.lockExclusive();
				
				KJ_IF_MAYBE(pDontCare, *locked) {
				} else {
					*locked = *pErr;
				}
			}
		};
		
		auto whenDone = [fulfiller, ctx]() mutable {
			auto locked = ctx -> exception.lockExclusive();
			
			KJ_IF_MAYBE(pErr, *locked) {
				fulfiller.reject(*pErr);
			} else {
				fulfiller -> fulfill();
			}
		};
		
		/*auto result = newOperation();
		
		struct DoneFunctor {
			Own<const Operation> op;
			
			DoneFunctor(Own<const Operation> op) :
				op(mv(op))
			{}
			
			DoneFunctor(const DoneFunctor& other) :
				op(other.op->addRef())
			{}
			
			void operator()() {
				op->done();
			}
		};
		
		auto funcPtr = kj::heap<decltype(func)>(mv(func));
		auto funcCopyable = [ptr = funcPtr.get()](Eigen::Index start, Eigen::Index end) {
			(*ptr)(start, end);
		};
		result->attachDestroyAnywhere(mv(funcPtr));
		
		device.parallelForAsync(n, cost, funcCopyable, DoneFunctor(result->addRef()));*/
		device.parallelForAsync(n, cost, func, whenDone);
		
		return mv(paf.promise);
	}
};

#ifdef FSC_WITH_CUDA

template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Kernel, Kernel func, typename... Params>
	static Own<Operation> launch(Eigen::GpuDevice& device, size_t n, const Eigen::TensorOpCost& cost, Params... params) {
		internal::gpuLaunch<Kernel, func, Params...>(device, n, params...);
		
		auto streamStatus = cudaStreamQuery(device.stream());
		KJ_REQUIRE(streamStatus == cudaSuccess || streamStatus == cudaErrorNotReady, "CUDA launch failed", streamStatus, cudaGetErrorName(streamStatus), cudaGetErrorString(streamStatus));
		
		auto op = newOperation();
		synchronizeGpuDevice(device, *op);
		return op;
		/*return prerequisite.then([&device, n, cost, params...]() {
			KJ_LOG(WARNING, "Launching GPU kernel");
			internal::gpuLaunch<Kernel, func, Params...>(device, n, params...);
			
			auto streamStatus = cudaStreamQuery(device.stream());
			KJ_REQUIRE(streamStatus == cudaSuccess || streamStatus == cudaErrorNotReady, "CUDA launch failed", streamStatus, cudaGetErrorName(streamStatus), cudaGetErrorString(streamStatus));
		});*/
	}
};

#endif // FSC_WITH_CUDA

// MappedData
	
template<typename T, typename Device>
MappedData<T, Device>::MappedData(Device& device, T* hostPtr, NonConst* devicePtr, size_t size) :
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
	devicePtr(deviceAlloc(device, size)),
	size(size)
{
}

template<typename T, typename Device>
MappedData<T, Device>::MappedData(Device& device) :
	device(device),
	hostPtr(nullptr),
	devicePtr(nullptr),
	size(0)
{
}


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
		unwindDetector.catchExceptionsIfUnwinding([&, this]() {
			device.deallocate(devicePtr);
			// hostMemSynchronizeBlocking(device);
		});
	}
}

template<typename T, typename Device>
void MappedData<T, Device>::updateHost() {
	KJ_REQUIRE(!kj::isConst<T>(), "Can not update host on a const type") {
		return;
	}
	device.memcpyDeviceToHost(const_cast<NonConst*>(hostPtr), devicePtr, size * sizeof(T));
}

template<typename T, typename Device>
void MappedData<T, Device>::updateDevice() {
	device.memcpyHostToDevice(devicePtr, hostPtr, size * sizeof(T));
}

template<typename T, typename Device>
RemoveConst<T>* MappedData<T, Device>::deviceAlloc(Device& device, size_t size) {
	return (NonConst*) device.allocate(size * sizeof(T));
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