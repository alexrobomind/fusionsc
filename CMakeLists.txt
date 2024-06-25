cmake_minimum_required(VERSION 3.20)

# Configure project

project(FSC LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
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
include(GNUInstallDirs)

# Pre-load python to make sure we have compatible dev environment and interpreter
#set(CMAKE_FIND_DEBUG_MODE On)
find_package(Python COMPONENTS Interpreter Development.Module NumPy)

# Try loading OpenMP
find_package(OpenMP)

message(STATUS "Python sitearch dir: ${Python_SITEARCH}")
#set(CMAKE_FIND_DEBUG_MODE Off)

if(${Python_Development.Module_FOUND} AND ${Python_NumPy_FOUND})
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

# Configure CCache if available
#find_program(CCACHE_LOC ccache)
#if(CCACHE_LOC)
#	message(STATUS "ccache found at ${CCACHE_LOC}")
#	
#	set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_LOC})
#	set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_LOC})
#else()
#	message(STATUS "ccache not found")
#endif()

include(src/cmake/AddVendoredDependency.cmake)

find_package(OpenSSL COMPONENTS Crypto SSL)

# Disable BUILD_TESTING while compiling dependencies
set(BUILD_TESTING_TMP ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "Temporarily disable testing" FORCE)

set(BUILD_EXAMPLES_TMP ${BUILD_EXAMPLES})
set(BUILD_EXAMPLES OFF CACHE BOOL "Temporarily disable examples" FORCE)

# Standard fetching for Catch2, capnproto and eigen
#CPMAddPackage(
#  Catch2
#  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
#  GIT_TAG        devel
#)

add_vendored_dependency(
	PREFIX vendor/catch2
	NAME Catch2
	VERSION 3.0
)

# We need to include the test scan macro manually if we downloaded it manually
if(Catch2_VENDORED)
	list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
else()
	list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()

set(SKIP_INSTALL_ALL On)
add_vendored_dependency(
	PREFIX vendor/zlib
	NAME ZLIB
	VERSION 1.2.0
)
unset(SKIP_INSTALL_ALL)

#CPMAddPackage(
#	NAME ZLIB
#	GIT_REPOSITORY https://github.com/madler/zlib.git
#	GIT_TAG        v1.2.13
#	VERSION        1.2.0
#)

if(ZLIB_VENDORED)
	set(ZLIB_CUSTOM_INCLUDES $<BUILD_INTERFACE:${ZLIB_SOURCE_DIR}> $<BUILD_INTERFACE:${ZLIB_BINARY_DIR}>)
	target_include_directories(zlib PUBLIC ${ZLIB_CUSTOM_INCLUDES})
	target_include_directories(zlibstatic PUBLIC ${ZLIB_CUSTOM_INCLUDES})
	
	message(STATUS "ZLib includes: ${ZLIB_CUSTOM_INCLUDES}")

	add_library(ZLIB::zlibstatic ALIAS zlibstatic)
	add_library(ZLIB::zlib ALIAS zlib)

	if(NOT TARGET ZLIB::ZLIB)
		add_library(ZLIB::ZLIB ALIAS zlibstatic)
	endif()
	
	target_include_directories(zlibstatic INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
	INSTALL(TARGETS zlibstatic EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
	INSTALL(DIRECTORY "${ZLIB_SOURCE_DIR}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
else()
	add_library(ZLIB::zlib ALIAS ZLIB::ZLIB)
	add_library(ZLIB::zlibstatic ALIAS ZLIB::ZLIB)
endif()
set(ZLIB_FOUND 1)

set(ZLIB_STATIC_LIBRARY ZLIB::zlibstatic)
set(ZLIB_LIBRARY ZLIB::zlib)
set(ZLIB_INCLUDE_DIRECTORIES "")
set(ZLIB_INCLUDE_DIR "")

add_vendored_dependency(
	PREFIX vendor/sqlite3
	NAME SQLite3
	VERSION 3.9.0
)

set(WITH_ZLIB OFF)
add_vendored_dependency(
	PREFIX vendor/capnproto
	NAME CapnProto
	VERSION 0.11
)
unset(WITH_ZLIB)

# Enable ZLIB support for vendored Capnproto
if(CapnProto_VENDORED)
	target_compile_definitions(kj-http PRIVATE KJ_HAS_ZLIB)
	target_link_libraries(kj-http PUBLIC ZLIB::ZLIB)
	
	if(NOT TARGET CapnProto::capnpc)
		add_library(CapnProto::capnpc ALIAS capnpc)
	endif()
endif()

add_vendored_dependency(
	PREFIX vendor/yaml-cpp
	NAME yaml-cpp
	VERSION 0.7
)

if(yaml-cpp_VENDORED)
	INSTALL(TARGETS yaml-cpp EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
	# Pre 0.7.1-versions do not export the yaml-cpp target with a namespace
	# This adds an alias to fix
	if(NOT TARGET yaml-cpp::yaml-cpp)
		add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
	endif()
endif()

#CPMAddPackage(
#  NAME CapnProto
#  GIT_REPOSITORY https://github.com/capnproto/capnproto.git
#  GIT_TAG        master
#  VERSION        0.11
#  OPTIONS
#    "WITH_ZLIB OFF"
#)

# We use Capn'n'protos builtin install rules

# Cap'n'proto provides a great macro to configure compilation of its interface files
# This macro either uses the target capnp_tool (for self-build) or the CAPNP_INCLUDE_DIRECTORY
# variable (when configured by find_package) to define its reference directories.
# However, when using a find_package call through CPMAddPackage, the CAPNP_INCLUDE_DIRECTORY
# does not get propagated to this scope. Therefore, we have to extract it manually.

#if(CapnProto_DIR)
#	get_property(CAPNP_INCLUDE_DIRECTORY TARGET CapnProto::capnp PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
#endif()

add_vendored_dependency(
	PREFIX vendor/eigen3
	NAME Eigen3
	VERSION 3.4
)

set(FSC_WITH_HDF5 On)
set(HDF5_EXTERNALLY_CONFIGURED 1)
set(H5_ZLIB_HEADER "zlib.h")
#set(HDF5_LIB_DEPENDENCIES ZLIB::ZLIB)
set(HDF5_EXPORTED_TARGETS "FSCTargets")
set(LINK_COMP_LIBS ZLIB::ZLIB)

set(HDF5_BUILD_CPP_LIB ON CACHE BOOL "description" FORCE)

add_vendored_dependency(
	PREFIX vendor/hdf5
	NAME HDF5
	VERSION 1.14.2
	FIND_ARGS COMPONENTS CXX HL
)

if(HDF5_VENDORED)
	add_library(hdf5::hdf5 ALIAS hdf5-static)
	add_library(hdf5::hdf5_cpp ALIAS hdf5_cpp-static)
	add_library(hdf5::hdf5_hl ALIAS hdf5_hl-static)
	add_library(hdf5::hdf5_hl_cpp ALIAS hdf5_hl_cpp-static)
endif()

# NetCDF requires a pre-built HDF5 to work, so we can't package it in here :/
#CPMAddPackage(
#	NAME netCDF
#	GIT_REPOSITORY https://github.com/Unidata/netcdf-c
#	GIT_TAG        v4.9.0
#)

add_vendored_dependency(
	PREFIX vendor/libssh2
	NAME Libssh2
	VERSION 1.10
)

	

#CPMAddPackage(
#	NAME Libssh2
#	GIT_REPOSITORY https://github.com/libssh2/libssh2
#	GIT_TAG        libssh2-1.10.0
#	VERSION        1.10.0
#)

if(Libssh2_VENDORED)
	add_library(Libssh2::libssh2 ALIAS libssh2)
	target_compile_definitions(libssh2 PRIVATE LIBSSH2DEBUG)
	
	# TODO: Install rules?
endif()

if(FSC_WITH_PYTHON)
	add_vendored_dependency(
		PREFIX vendor/pybind11
		NAME pybind11
		VERSION 2.9.1
	)
	#CPMAddPackage(
	#	NAME pybind11
	#	GIT_REPOSITORY https://github.com/pybind/pybind11.git
	#	GIT_TAG        v2.9.1
	#	VERSION        2.9.1
	#)
endif()

add_vendored_dependency(
	PREFIX vendor/botan
	NAME Botan
	VERSION 2.9
	SETUP_ONLY
)

if(Botan_VENDORED)
	include(src/cmake/CompileBotan.cmake)
	
	INSTALL(TARGETS botan_selfbuilt EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

	get_property(BOTAN_HEADERS TARGET botan_selfbuilt PROPERTY BUILT_HEADERS)
	INSTALL(DIRECTORY "${BOTAN_HEADERS}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
	
# For Botan, we have to be fancier, as it does not use CMake
#CPMAddPackage(
#	NAME Botan
#	GIT_REPOSITORY https://github.com/randombit/botan.git
#	GIT_TAG 2.19.0
#	VERSION 2.19.0
#	DOWNLOAD_ONLY
#)

add_library(fusionsc_jsonscons INTERFACE)
target_include_directories(fusionsc_jsonscons INTERFACE vendor/jsonscons/include)

# Re-enable BUILD_TESTING if desired
set(BUILD_TESTING ${BUILD_TESTING_TMP} CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES ${BUILD_EXAMPLES_TMP} CACHE BOOL "" FORCE)

add_subdirectory(vendor/capfuzz)

add_subdirectory(vendor/happly)

add_library(fusionsc_poissongen INTERFACE)
target_include_directories(fusionsc_poissongen INTERFACE vendor/pdg)

add_library(deps INTERFACE)
target_link_libraries(
	deps
	INTERFACE
	Botan::botan
	CapFuzz::capfuzz
	CapnProto::capnpc
	CapnProto::capnp-rpc
	CapnProto::capnp-json
	CapnProto::capnp-websocket
	CapnProto::kj-http
	CapnProto::kj
	Eigen3::Eigen
	happly::happly
	hdf5::hdf5_cpp
	hdf5::hdf5_hl_cpp
	Libssh2::libssh2
	SQLite::SQLite3
	ZLIB::ZLIB
	yaml-cpp::yaml-cpp
)

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

option(FSC_SANITIZE_ADDRESS "Whether to enable address sanitizer")

if(FSC_SANITIZE_ADDRESS)
	target_compile_options(deps INTERFACE "-fsanitize=address")
	target_link_options(deps INTERFACE "-fsanitize=address")
	
	if(CapnProto_VENDORED)
		target_compile_options(kj PUBLIC "-fsanitize=address")
		target_link_options(kj PUBLIC "-fsanitize=address")
	endif()
endif()

option(FSC_ARCH_NATIVE "Whether to build for host architecture")

if(FSC_ARCH_NATIVE)
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
	
	if(COMPILER_SUPPORTS_MARCH_NATIVE)
		target_compile_options(deps INTERFACE "-march=native")
		set(FSC_ARCH_NATIVE_RESULT "Enabled")
	else()
		set(FSC_ARCH_NATIVE_RESULT "Unsupported by compiler")
	endif()
else()
	set(FSC_ARCH_NATIVE_RESULT "Disabled")
endif()

option(FSC_WITH_OPENMP "Whether to enable OpenMP support" On)

# Set up the scikit build
if(SKBUILD)
	add_custom_target(fsc-skbuild-target)
	set(FSC_WITH_OPENMP Off CACHE BOOL "OpenMP disabled in scikit build")
endif()

# Set up testing subsystem
include(Catch)

# First build the cupnp library / compiler
add_subdirectory(src/c++/cupnp)
add_subdirectory(src/c++/capnpc-java)
add_subdirectory(src/c++/capnp-sphinx)

# Build and install the libraries
add_subdirectory(src/python/fusionsc/serviceDefs/fusionsc) # This directory contains the service interface
add_subdirectory(src/c++)

# Build the documentation
add_subdirectory(src/docs)

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

install(EXPORT FSCTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake NAMESPACE FSC::)
export(EXPORT FSCTargets NAMESPACE fsc FILE FSCExports.cmake)

# Print information about build
message(STATUS "")
message(STATUS "--- Configuration options ---")
message(STATUS "Python support: ${FSC_WITH_PYTHON}")
message(STATUS "CUDA support: ${FSC_WITH_CUDA}")
message(STATUS "OpenMP support: ${OPENMP_CXX_FOUND}")
message(STATUS "OpenMP usage: ${FSC_WITH_OPENMP}")
message(STATUS "Native Arch support: ${FSC_ARCH_NATIVE_RESULT}")
