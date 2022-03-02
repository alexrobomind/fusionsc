#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define CUPNP_GPUCC
#endif

#ifdef __CUDA__
	#include <cuda.h>
#endif

namespace cupnp {
	#if defined __CUDACC__
		auto gpuMalloc(void** target, size_t size) { return cudaMalloc(target, size); }
		auto gpuFree(void* target) { return cudaFree(target); }
		
		inline void memcpyToDevice(void* dst, const void* src, size_t size) {
			auto err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
			KJ_REQUIRE(err == 0, "cudaMemcpy failure (host -> device)");
		}
		
		inline void memcpyToHost(void* dst, const void* src, size_t size) {
			auto err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
			KJ_REQUIRE(err == 0, "cudaMemcpy failure (device -> host)");
		}		
	#endif
	
	#if defined(CUPNP_GPUCC)
		namespace internal {
		// Implementation of kj::Array for GPU devices
			struct DeviceArrayDisposer : public kj::ArrayDisposer {
				const static inline DeviceArrayDisposer instance;
				
				inline void disposeImpl(
					void* firstElement, size_t elementSize, size_t elementCount,
					size_t capacity, void (*destroyElement)(void*)
				) const override {
					auto err = gpuFree(firstElement);
					
					if(err != 0) {
						KJ_LOG(WARNING, "Error when freeing device array", err);
					}
				}
			};
		}
		
		template<typename T>
		kj::Array<T> deviceArray(size_t size) {
			void* ptr = nullptr;
			auto err = gpuMallog(&ptr, size * sizeof(T));
			KJ_REQUIRE(err == 0, "Device allocation failure");
			
			return kj::Array<T>(ptr, size, internal::DeviceArrayDisposer::instance);
		}
	/*# else
		template<typename T>
		kj::Array<T> deviceArray(size_t size) {
			return kj::heapArray<T>(size);
		}
		
		inline void memcpyToDevice(void* dst, const void* src, size_t size) {
			memcpy(dst, src, size);
		}
		
		inline void memcpyToHost(void* dst, const void* src, size_t size) {
			memcpy(dst, src, size);
		}*/
	# endif
}
