option(FSC_WITH_AVX "Whether to compile FSC (and used dependencies) with AVX instructions" OFF)
option(FSC_WITH_AVX2 "Whether to compile FSC (and used dependencies) with AVX2 instructions" OFF)
option(FSC_NATIVE, "Whether to compile FSC for host processor" OFF)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(AVX_FLAGS "-mavx")
	set(AVX2_FLAGS "-mavx2")
	set(NATIVE_FLAGS "-march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(AVX_FLAGS "-mavx")
	set(AVX2_FLAGS "-mavx2")
	set(NATIVE_FLAGS "-march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(AVX_FLAGS "-mavx")
	set(AVX2_FLAGS "-mavx2")
	set(NATIVE_FLAGS "-march=native")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
	set(AVX_FLAGS "/arch:AVX")
	set(AVX2_FLAGS "/arch:AVX2")
	set(NATIVE_FLAGS "")
endif()

if(FSC_WITH_AVX)
	add_compile_options(${AVX_FLAGS})
endif()

if(FSC_WITH_AVX2)
	add_compile_options(${AVX2_FLAGS})
endif()

if(FSC_NATIVE)
	# Check if we have AVX available
	include(CheckSourceRuns)
	
	set(CMAKE_REQUIRED_FLAGS ${AVX_FLAGS})
	check_source_runs(CXX "
		#include <immintrin.h>
		
		int main() {
			__m256 var;
			var = _mm256_set1_ps(9.63);
			var = _mm256_add_ps(var,var);
			return 0;
		}
	" HOST_HAS_AVX)
	
	if(HOST_HAS_AVX)
		add_compile_options(${AVX_FLAGS})
		message(STATUS "Adding AVX flags through FSC_NATIVE option")
	endif()
	
	set(CMAKE_REQUIRED_FLAGS ${AVX2_FLAGS})
	check_source_runs(CXX "
		#include <immintrin.h>
		
		int main() {
			__m256i var;
			var = _mm256_set_epi64x(1, 2, 3, 4);
			var = _mm256_sub_epi64(var,var);
			return 0;
		}
	" HOST_HAS_AVX2)
	
	if(HOST_HAS_AVX2)
		add_compile_options(${AVX2_FLAGS})
		message(STATUS "Adding AVX2 flags through FSC_NATIVE option")
	endif()
	
	add_compile_options(${NATIVE_FLAGS})
endif()