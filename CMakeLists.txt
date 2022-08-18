cmake_minimum_required(VERSION 3.17)

# Dependencies:
# - capnproto (https://github.com/capnproto/capnproto)
# - Botan (https://github.com/randombit/botan)
# - Catch2 (https://github.com/catchorg/Catch2)
# - Eigen3 (https://gitlab.com/libeigen/eigen.git)
# - pybind11 (https://github.com/pybind/pybind11.git)

# Configure project

project(FSC LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/cmake/modules)

# Configure languages

# Try to enable CUDA
option(FSC_WITH_CUDA "Whether to use CUDA compilation")
if(FSC_WITH_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit)
endif()

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)
option(BUILD_STATIC_LIBS "Build Static Libraries" ON)

set(CMAKE_CXX_EXTENSIONS On)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(src/cmake/OptimizeForTarget.cmake)

# Prepare CTest
include(CTest)

# Pre-load python to make sure we have compatible dev environment and interpreter
find_package(Python COMPONENTS Interpreter Development NumPy)

if(${Python_Development_FOUND} AND ${Python_NumPy_FOUND})
	set(FSC_WITH_PYTHON ON)
else()
	set(FSC_WITH_PYTHON OFF)
endif()

if(SKBUILD AND NOT Python_FOUND)
	message(FATAL_ERROR "Could not find python in python-driven build")
endif()

if(SKBUILD AND NOT Python_NumPy_FOUND)
	message(FATAL_ERROR "NumPy missing in SKBUILD build")
endif()

# Try to load dependencies directly

set(FSC_FETCH_DEPENDENCIES On CACHE BOOL "Whether the build system is allowed to fetch external dependencies")
set(FSC_ALWAYS_FETCH On CACHE BOOL "Whether the build system must fetch external dependencies")

if(NOT FSC_ALWAYS_FETCH OR NOT FSC_FETCH_DEPENDENCIES)
	find_package(CapnProto CONFIG)
	find_package(Botan MODULE)
	find_package(Catch2 CONFIG)
	find_package(Eigen3 CONFIG)
	find_package(pybind11 CONFIG)
	find_package(HDF5 CONFIG)
else()
	set(CapnProto_FOUND FALSE)
	set(Botan_FOUND FALSE)
	set(Catch2_FOUND FALSE)
	set(Eigen3_FOUND FALSE)
	set(pybind11_FOUND FALSE)
	set(HDF5_FOUND FALSE)
endif()

# Try to fetch some dependencies if possible

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        devel
  GIT_SHALLOW    On
  GIT_PROGRESS   On
)

FetchContent_Declare(
  CapnProto
  #GIT_REPOSITORY https://github.com/capnproto/capnproto.git
  #GIT_TAG        v0.9.1
  GIT_REPOSITORY https://github.com/alexrobomind/capnproto.git
  GIT_TAG        schema_list_scope_bindings
  GIT_SHALLOW    On
  GIT_PROGRESS   On
)

FetchContent_Declare(
  Botan
  GIT_REPOSITORY https://github.com/randombit/botan.git
  GIT_TAG        2.18.2
  GIT_SHALLOW    On
  GIT_PROGRESS   On
)

FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://github.com/alexrobomind/eigen34.git
  GIT_TAG        3.4
  #GIT_REPOSITORY https://github.com/alexrobomind/eigen.git
  #GIT_TAG        cudafix
  GIT_SHALLOW    On
  GIT_PROGRESS   On
)

FetchContent_Declare(
	pybind11
	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	GIT_TAG        v2.9.1
	GIT_SHALLOW    On
	GIT_PROGRESS   On
)

FetchContent_Declare(
	HDF5
	GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
	GIT_TAG        hdf5-1_13_2
	GIT_SHALLOW    On
	GIT_PROGRESS   On
)
	

# Disable BUILD_TESTING while compiling dependencies
set(BUILD_TESTING_TMP ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "Temporarily disable testing" FORCE)

macro(fetch_dependency DEPNAME)
	if(NOT ${DEPNAME}_FOUND)
		if(NOT FSC_FETCH_DEPENDENCIES)
			message(FATAL_ERROR "${DEPNAME} could not be found, and dependencies fetching was disabled")
		endif()
		
		message(STATUS "${DEPNAME} not found. Fetching from repository")
		FetchContent_MakeAvailable(${DEPNAME})
	endif()
endmacro()

# Standard fetching for Catch2, capnproto and eigen
fetch_dependency(Catch2)

# We need to include the test scan macro manually if we downloaded it manually
if(NOT Catch2_FOUND)
	list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
endif()

fetch_dependency(CapnProto)
fetch_dependency(Eigen3)

# HDF5's shared library option messes with other shared library builds
SET(HDF5_EXTERNALLY_CONFIGURED 1)
SET(HDF5_BUILD_HL_LIB ON CACHE BOOL "Enable building of HDF5 HL library" FORCE)

fetch_dependency(HDF5)

if(FSC_WITH_PYTHON)
	fetch_dependency(pybind11)
endif()

# For Botan, we have to be fancier, as it does not use CMake
if(NOT Botan_FOUND)
	if(NOT FSC_FETCH_DEPENDENCIES)
		message(FATAL_ERROR "Botan could not be found, and dependency fetching was disabled")
	endif()
	
	include(src/cmake/CompileBotan.cmake)
endif()

# Re-enable BUILD_TESTING if desired
set(BUILD_TESTING ${BUILD_TESTING_TMP} CACHE BOOL "" FORCE)

add_library(deps INTERFACE)
target_link_libraries(
	deps
	INTERFACE
	Botan::botan
	CapnProto::capnp-rpc
	CapnProto::capnp-json
	CapnProto::kj-http
	CapnProto::kj
	Eigen3::Eigen
)

if(TARGET capnpc)
	target_link_libraries(deps INTERFACE capnpc)
else()
	target_link_libraries(deps INTERFACE CapnProto::capnpc)
endif()

if(TARGET hdf5-static)
	target_link_libraries(deps INTERFACE hdf5-static)
else()
	target_link_libraries(deps INTERFACE HDF5::hdf5-static)
endif()


if(FSC_WITH_CUDA)
	target_compile_definitions(
		deps
		INTERFACE
		FSC_WITH_CUDA
	)
	target_link_libraries(
		deps
		INTERFACE
		CUDA::cudart
	)
endif()

if(FSC_WITH_PYTHON)
	target_compile_definitions(
		deps
		INTERFACE
		FSC_WITH_PYTHON
	)
	target_link_libraries(
		deps
		INTERFACE
		pybind11::pybind11
		Python::NumPy
	)
endif()

# Set up testing subsystem
include(Catch)

# Cross-communication variables
SET(FSC_GENAPI "")

# Build and install the libraries
add_subdirectory(src/c++)

# Build the documentation
add_subdirectory(docs)

# Coverage testing
option(FSC_WITH_INSTRUMENTATION "Whether to perform instrumentation")
if(FSC_WITH_INSTRUMENTATION AND (CMAKE_CXX_COMPILER_ID STREQUAL "Clang") AND BUILD_TESTING)

	target_compile_options(fsc PRIVATE -fprofile-instr-generate -fcoverage-mapping)
	target_link_options(tests PRIVATE -fprofile-instr-generate -fcoverage-mapping)
	
	set(LLVM_PROFDATA_COMMAND "llvm-profdata" CACHE STRING "Path to llvm-profdata executable")
	
	add_custom_command(
		OUTPUT fsc.profraw
		COMMAND "${CMAKE_COMMAND}" ARGS -E env LLVM_PROFILE_FILE=fsc.profraw $<TARGET_FILE:tests>
		DEPENDS tests
	)
	
	add_custom_command(
		OUTPUT fsc.profdata
		COMMAND "${LLVM_PROFDATA_COMMAND}" ARGS merge -sparse fsc.profraw -o fsc.profdata
		DEPENDS fsc.profraw
	)
	
	add_custom_target(
		profiledata
		DEPENDS fsc.profdata
	)
endif()

# Extra target for the python-driven build
if(SKBUILD)
	add_custom_target(fsc-install-skbuild
		${CMAKE_COMMAND}
		-DCMAKE_INSTALL_COMPONENT=SKBUILD
		-P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
		DEPENDS native
    )
endif(SKBUILD)