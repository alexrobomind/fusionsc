#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
	#define CUPNP_GPUCC
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	#define CUPNP_DEVICE_COMPILATION_PHASE
#endif

#if defined(CUPNP_GPUCC) && !defined(CUPNP_GPU_COMPILATION_PHASE)
	#define CUPNP_HOST_COMPILATION_PHASE
#endif

#ifdef CUPNP_WITH_CUDA
	#include <cuda.h>
#endif

# ifdef CUPNP_GPUCC
	//TODO: Better error handling for device code?
	# define CUPNP_REQUIRE(expr, ...) if(!(expr)) [[unlikely]]
	# define CUPNP_FAIL_REQUIRE(...) if(true)
	# define CUPNP_DBG(...)
# else
	# include <kj/debug.h>

	# define CUPNP_REQUIRE(...) KJ_REQUIRE((__VA_ARGS__))
	# define CUPNP_FAIL_REQUIRE(...) KJ_FAIL_REQUIRE((__VA_ARGS__))
	# define CUPNP_DBG(...) KJ_DBG(__VA_ARGS__)
# endif

#ifdef CUPNP_GPUCC
	#define CUPNP_FUNCTION __host__ __device__
#else
	#define CUPNP_FUNCTION
#endif

#include <cstdlib>

namespace cupnp {
	# ifdef CUPNP_WITH_HIP
		inline auto deviceMalloc(void** target, size_t size) { return hipMalloc(target, size); }
		inline auto deviceFree(void* target) { return hipFree(target); }
	#else
		# ifdef CUPNP_WITH_CUDA
			inline auto deviceMalloc(void** target, size_t size) { return cudaMalloc(target, size); }
			inline auto deviceFree(void* target) { return cudaFree(target); }
		# else
			inline int deviceMalloc(void** target, size_t size) { *target = malloc(size); return 0; }
			inline int deviceFree(void* target) { free(target); return 0; }
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
	
	template<typename T>
	auto deviceMemcpy(kj::ArrayPtr<T> dst, const kj::ArrayPtr<const T> src) {
		CUPNP_REQUIRE(dst.size() >= src.size());
		const auto nBytes = src.size() * sizeof(T);
		
		return deviceMemcpy(dst.begin(), src.begin(), nBytes);
	}
	
	template<typename T>
	auto deviceMemcpy(kj::ArrayPtr<T> dst, const kj::ArrayPtr<T> src) {
		return deviceMemcpy(dst, src.asConst());
	}
	
	namespace internal {
		// Implementation of kj::Array for GPU devices
		struct DeviceArrayDisposer final : public kj::ArrayDisposer {			
			inline DeviceArrayDisposer() = default;
			
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
		
		const inline DeviceArrayDisposer DEVICE_ARRAY_DISPOSTER_INSTANCE;
	}
	
	template<typename T>
	kj::Array<T> deviceArray(size_t size) {
		void* ptr = nullptr;
		auto err = deviceMalloc(&ptr, size * sizeof(T));
		KJ_REQUIRE(err == 0, "Device allocation failure");
		
		return kj::Array<T>(reinterpret_cast<T*>(ptr), size, internal::DEVICE_ARRAY_DISPOSTER_INSTANCE);
	}
}
