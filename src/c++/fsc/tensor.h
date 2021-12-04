#pragma once

# include <thread>
# include <kj/debug.h>
# include <kj/async.h>
# include <kj/memory.h>

# ifdef _OPENMP
	# include <omp.h>
# endif

#include "common.h"
#include "tensor-local.h"

namespace fsc {
	
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
	
	template<typename T, int rank, int options, typename Index, typename T2>
	void readTensor(T2 reader, Tensor<T, rank, options, Index>& out);
	
	template<typename T, typename T2>
	T readTensor(T2 reader);

	template<typename T, int rank, int options, typename Index, typename T2>
	void writeTensor(const Tensor<T, rank, options, Index>& in, T2 builder);
	
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
		auto func = [f = mv(f), params...](Eigen::Index start, Eigen::Index end) mutable {
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
		
		return kj::evalLater([funcCopyable, doneCopyable, cost, n, &device, promise = mv(paf.promise)]() mutable {
			device.parallelForAsync(n, cost, funcCopyable, doneCopyable);
			return mv(promise);
		}).attach(mv(donePtr), mv(funcPtr));
	}
};

#ifdef EIGEN_USE_GPU

namespace internal {
	
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
		
		fulfiller->reject(KJ_EXCEPTION(FAILED, "GPU computation failed", status));
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
	auto fulfiller = new Own<kj::CrossThreadPromiseFulfiller<void>>(); // delete in internal::gpuSynchCallback
	*fulfiller = mv(paf.fulfiller);
	
	# ifdef __CUDACC__
		auto result = cudaStreamAddCallback(device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# elif HIP
		auto result = hipStreamAddCallback (device.stream(), internal::gpuSynchCallback, (void*) fulfiller, 0);
	# endif
	
	// If the operation failed, we can't trust the callback to be called
	// Better just fail now
	if(result != gpuSuccess) {
		// We don't know for sure whether the function will be called or not
		// Better eat a tiny memory leak than calling undefined behavior (so no delete)
		*fulfiller = nullptr;
		KJ_FAIL_REQUIRE("Callback scheduling returned error code", result);
	}
	
	return mv(paf.promise);
}

template<>
struct KernelLauncher<Eigen::GpuDevice> {
	template<typename Func, typename... Params>
	static Promise<void> launch(Eigen::GpuDevice& device, Func f, size_t n, Eigen::TensorOpCost& cost, Params... params) {
		#ifndef EIGEN_GPUCC
			static_assert(sizeof(Func) == 0, "KernelLauncher<Eigen::GpuDevice>::launch must be called from within a GPU compiler (e.g. HIP / Cuda)");
		#else
			//#ifdef EIGEN_GPU_COMPILE_PHASE
			//	static_assert(sizeof(Func) == 0, "KernelLauncher<Eigen::GpuDevice>::launch may not be called from device-side code");
			//#else
				// LAUNCH_GPU_KERNEL(f, n, 1, 0, device, params...);
				gpuLaunch(device, f, n, params...);
				return synchronizeGpuDevice(device);
			//#endif
		#endif
	}
};

#endif
	
template<typename T, int rank, int options, typename Index, typename T2>
void readTensor(T2 reader, Tensor<T, rank, options, Index>& out) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = reader.getShape();
		KJ_REQUIRE(out.rank() == shape.size());
	
		typename TensorType::Dimensions dims;
		for(size_t i = 0; i < rank; ++i) {
			if(options & Eigen::RowMajor) {
				dims[i] = shape[i];
			} else {
				dims[i] = shape[rank - i - 1];
			}
		}
	
		out.resize(dims);
	}
	
	auto data = reader.getData();
	KJ_REQUIRE(out.size() == data.size());
	auto dataOut = out.data();
	for(size_t i = 0; i < out.size(); ++i)
		dataOut[i] = (T) data[i];
}

template<typename T, typename T2>
T readTensor(T2 reader) {
	T result;
	readTensor(reader, result);
	return mv(result);
}
	
template<typename T, int rank, int options, typename Index, typename T2>
void writeTensor(const Tensor<T, rank, options, Index>& in, T2 builder) {
	using TensorType = Tensor<T, rank, options, Index>;
	
	{
		auto shape = builder.initShape(rank);
	
		auto dims = in.dimensions();
		for(size_t i = 0; i < rank; ++i) {
			if(options & Eigen::RowMajor) {
				shape.set(i, dims[i]);
			} else {
				shape.set(rank - i - 1, dims[i]);
			}
		}
	}
	
	auto dataOut = builder.initData(in.size());
	auto data = in.data();
	for(size_t i = 0; i < in.size(); ++i)
		dataOut.set(i, data[i]);
}

}