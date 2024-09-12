namespace fsc {

namespace internal {					
	/**
	 * Helper function that actually launches the kernel. Needed to perform index expansion
	 * over the parameter.
	 */
	template<typename Kernel, Kernel f, typename Device, typename... Params, size_t... i>
	Promise<void> auxKernelLaunch(Device& device, size_t n, Promise<void> prerequisite, Eigen::TensorOpCost cost, std::index_sequence<i...> indices, Params&&... params) {
		if(n == 0) return READY_NOW;
		
		// Create mappers for input
		auto mappers = heapHeld<kj::Tuple<DeviceMappingType<Params>...>>(kj::tuple(
			mapToDevice(fwd<Params>(params), device, true)...
		));
					
		using givemeatype = int[];
		
		auto paf = kj::newPromiseAndFulfiller<void>();
		
		Promise<void> result = prerequisite.then([&device, mappers]() mutable {
			// Update device memory
			// Note: This is an extremely convoluted looking way of calling updateDevice on all tuple members
			// C++ 17 has a neater way to do this, but we don't wanna require it just yet
			(void) (givemeatype { 0, (kj::get<i>(*mappers) -> updateDevice(), 0)... });
			return device.barrier();
		})
		.then([&device, cost, n, mappers, onCancel = mv(paf.promise)]() mutable {
			// Call kernel
			Promise<void> promise = KernelLauncher<Device>::template launch<Kernel, f, DeviceType<Params>...>(device, n, cost, mv(onCancel), kj::get<i>(*mappers) -> get()...);
			promise = promise.attach(device.addRef(), kj::get<i>(*mappers) -> addRef()...);
			return promise;
		}).then([&device, mappers]() mutable {
			// Update host memory
			(void) (givemeatype { 0, (kj::get<i>(*mappers) -> updateHost(), 0)... });
			return device.barrier();
		})
		.attach(mappers.x(), device.addRef());
		
		return getActiveThread().uncancelable(mv(result)).attach(mv(paf.fulfiller));
	}
}

/*
// Inline functions required to instantiate kernel launches

template<typename Device>
void potentiallySynchronize(Device& d) {}

#ifdef FSC_WITH_CUDA

template<>
inline void potentiallySynchronize<Eigen::GpuDevice>(Eigen::GpuDevice& d) {
	d.synchronize();
}

#endif*/

// =========================== CPU launcher ==============================

template<>
struct KernelLauncher<CPUDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(CPUDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> onCancel, Params... params) {
		auto paf = kj::newPromiseAndCrossThreadFulfiller<void>();
		AtomicShared<kj::CrossThreadPromiseFulfiller<void>> fulfiller = mv(paf.fulfiller);
		
		struct Context {
			kj::MutexGuarded<Maybe<kj::Exception>> exception;
			mutable std::atomic<bool> active = true;
		};
		AtomicShared<Context> ctx = kj::heap<Context>();
		AtomicShared<Tuple<Params...>> sharedParams = kj::heap<Tuple<Params...>>(kj::tuple(kj::fwd<Params>(params)...));
		
		onCancel = onCancel.attach(kj::defer([ctx]() { ctx -> active = false; }));
		
		// getActiveThread().detach(onCancel.catch_());
		
		auto func = [ctx, sharedParams](Eigen::Index start, Eigen::Index end) mutable {
			if(!ctx -> active.load())
				return;
			
			auto maybeException = kj::runCatchingExceptions([&sharedParams, start, end]() mutable {
				for(Eigen::Index i = start; i < end; ++i)
					//f(i, params...);
					kj::apply(f, i, *sharedParams);
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
				fulfiller -> reject(cp(*pErr));
			} else {
				fulfiller -> fulfill();
			}
		};
		device.eigenDevice().parallelForAsync(n, cost, func, whenDone);
		
		return paf.promise.attach(mv(onCancel));
	}
};

template<>
struct KernelLauncher<LoopDevice> {
	template<typename Kernel, Kernel f, typename... Params>
	static Promise<void> launch(LoopDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> onCancel, Params... params) {
		// Ignore cancelation
		(void) onCancel;
		
		for(size_t i : kj::range(0, n)) {
			f(i, params...);
		}
		
		return READY_NOW;
	}
};

	
#ifdef FSC_WITH_CUDA

// =========================== GPU launcher ==============================


template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Kernel, Kernel func, typename... Params>
	static Promise<void> launch(Eigen::GpuDevice& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> onCancel, Params... params) {
		// Ignore cancelation
		(void) onCancel;
		
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

// ========================= Generic launcher ============================

template<>
struct KernelLauncher<DeviceBase> {
	template<typename Kernel, Kernel func, typename... Params>
	static Promise<void> launch(DeviceBase& device, size_t n, const Eigen::TensorOpCost& cost, Promise<void> onCancel, Params... params) {
		#define FSC_HANDLE_TYPE(DevType) \
			if(device.brand == &DevType::BRAND) \
				return KernelLauncher<DevType>::launch<Kernel, func, Params...>(static_cast<DevType&>(device), n, cost, mv(onCancel), fwd<Params>(params)...);
				
		FSC_HANDLE_TYPE(CPUDevice);
		FSC_HANDLE_TYPE(LoopDevice);
		
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

}