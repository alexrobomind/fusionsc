#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define CUPNP_GPUCC
#endif

#ifdef CUPNP_WITH_CUDA
	#include <cuda.h>
#endif

namespace cupnp {
	# ifdef CUPNP_WITH_HIP
		auto deviceMalloc(void** target, size_t size) { return hipMalloc(target, size); }
		auto deviceFree(void* target) { return hipFree(target); }
	#else
		# ifdef CUPNP_WITH_CUDA
			auto deviceMalloc(void** target, size_t size) { return cudaMalloc(target, size); }
			auto deviceFree(void* target) { return cudaFree(target); }
		# else
			int deviceMalloc(void** target, size_t size) { *target = malloc(size); }
			int deviceFree(void* target) { free(target); }
		#endif
	#endif
		
	
	inline auto deviceMemcpy(void* dst, const void* src, size_t nBytes) {
		# ifdef CUPNP_WITH_HIP
			return hipMemcpy(dst, src, nBytes, cudaMemcpyDefault);
		# else
			# ifdef CUPNP_WITH_CUDA
				return cudaMemcpy(dst, src, nBytes, hipMemcpyDefault);
			# else
				memcpy(dst, src, nBytes);
				return (int) 0;
			# endif
		# endif
	}
	
	inline auto deviceMemcpy(kj::ArrayPtr<capnp::word> dst, const kj::ArrayPtr<const capnp::word> src) {
		CUPNP_REQUIRE(dst.size() >= src.size());
		const auto nBytes = src.size() * sizeof(capnp::word);
		
		return deviceMemcpy(dst.begin(), src.begin(), nBytes);
	}
	
	inline auto deviceMemcpy(kj::ArrayPtr<capnp::word> dst, const kj::ArrayPtr<capnp::word> src) {
		return deviceMemcpy(dst, src.asConst());
	}
	
	namespace internal {
		// Implementation of kj::Array for GPU devices
		struct DeviceArrayDisposer : public kj::ArrayDisposer {
			const static inline DeviceArrayDisposer instance;
			
			inline void disposeImpl(
				void* firstElement, size_t elementSize, size_t elementCount,
				size_t capacity, void (*destroyElement)(void*)
			) const override {
				auto err = deviceFree(firstElement);
				
				if(err != 0) {
					KJ_LOG(WARNING, "Error when freeing device array", err);
				}
			}
		};
	}
	
	template<typename T>
	kj::Array<T> deviceArray(size_t size) {
		void* ptr = nullptr;
		auto err = deviceMalloc(&ptr, size * sizeof(T));
		KJ_REQUIRE(err == 0, "Device allocation failure");
		
		return kj::Array<T>(ptr, size, internal::DeviceArrayDisposer::instance);
	}
}
