cmake_minimum_required(VERSION 3.17)

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

# Get dependencies via CPM
include(CPM.cmake)

find_package(OpenSSL COMPONENTS Crypto SSL)

# Disable BUILD_TESTING while compiling dependencies
set(BUILD_TESTING_TMP ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "Temporarily disable testing" FORCE)

set(BUILD_EXAMPLES_TMP ${BUILD_EXAMPLES})
set(BUILD_EXAMPLES OFF CACHE BOOL "Temporarily disable examples" FORCE)

# Standard fetching for Catch2, capnproto and eigen
CPMAddPackage(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        devel
)

# We need to include the test scan macro manually if we downloaded it manually
if(Catch2_SOURCE_DIR)
	list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
else()
	list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()

message(STATUS "Module path: ${CMAKE_MODULE_PATH}")

CPMAddPackage(
	NAME ZLIB
	GIT_REPOSITORY https://github.com/madler/zlib.git
	GIT_TAG        v1.2.13
)

message(STATUS "ZLIB added, ${ZLIB_DIR}")

if(TARGET zlib)
	message(STATUS "Local zlib")

	set(ZLIB_CUSTOM_INCLUDES $<BUILD_INTERFACE:${ZLIB_SOURCE_DIR}> $<BUILD_INTERFACE:${ZLIB_BINARY_DIR}>)
	target_include_directories(zlib PUBLIC ${ZLIB_CUSTOM_INCLUDES})
	target_include_directories(zlibstatic PUBLIC ${ZLIB_CUSTOM_INCLUDES})

	add_library(ZLIB::zlibstatic ALIAS zlibstatic)
	add_library(ZLIB::zlib ALIAS zlib)

	if(NOT TARGET ZLIB::ZLIB)
		add_library(ZLIB::ZLIB ALIAS zlibstatic)
	endif()

	# CapnProto needs this... Jesus ...
	# install(TARGETS zlibstatic EXPORT CapnProtoTargets)
else()
	find_package(ZLIB REQUIRED)

	add_library(ZLIB::zlib ALIAS ZLIB::ZLIB)
	add_library(ZLIB::zlibstatic ALIAS ZLIB::ZLIB)
endif()

set(ZLIB_STATIC_LIBRARY ZLIB::zlibstatic)
set(ZLIB_LIBRARY ZLIB::zlib)
set(ZLIB_INCLUDE_DIRECTORIES "")
set(ZLIB_INCLUDE_DIR "")

# set(CMAKE_FIND_DEBUG_MODE TRUE)
CPMAddPackage(
  NAME CapnProto
  GIT_REPOSITORY https://github.com/capnproto/capnproto.git
  GIT_TAG        master
  VERSION        REQUIRED
  OPTIONS
    "WITH_ZLIB OFF"
)

# A hack to extract the include directory
#if(CapnProto_DIR)
#	# Re-run find_package at global scope if found this way
#	find_package(CapnProto REQUIRED)
#endif()

#include(CMakePrintHelpers)
#cmake_print_properties(TARGETS CapnProto::capnp PROPERTIES LOCATION INTERFACE_INCLUDE_DIRECTORIES)

if(CapnProto_DIR)
	get_property(CAPNP_INCLUDE_DIRECTORY TARGET CapnProto::capnp PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
endif()
	

#if(NOT TARGET CapnProto::capnpc)
#	add_executable(CapnProto::capnpc ALIAS capnpc)
#endif()

CPMAddPackage(
  NAME Eigen3
  GIT_REPOSITORY https://github.com/alexrobomind/eigen34.git
  GIT_TAG        3.4
)

# HDF5's shared library option messes with other shared library builds
SET(HDF5_EXTERNALLY_CONFIGURED 1)
SET(HDF5_BUILD_HL_LIB ON CACHE BOOL "Enable building of HDF5 HL library" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT ON CACHE BOOL "Enable Hdf5 Zlib support" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "Enable HDF5 tools" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "Enable HDF5 examples" FORCE)

set(H5_ZLIB_HEADER "zlib.h")

CPMAddPackage(
	HDF5
	GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
	GIT_TAG        hdf5-1_13_2
)

if(NOT TARGET HDF5::hdf5-static)
	add_library(HDF5::hdf5-static ALIAS hdf5-static)
	add_library(HDF5::hdf5_hl-static ALIAS hdf5_hl-static)
endif()

#set(HDF5_C_LIBRARY HDF5::hdf5-static)
#set(HDF5_HL_LIBRARY HDF5::hdf5_hl-static)

CPMGetPackageVersion(HDF5 HDF5_VERSION)

#if(NOT HDF5_VERSION)
#	get_target_property(HDF5_INCDIRS HDF5::hdf5-static INCLUDE_DIRECTORIES)
#	
#	find_file (HDF5_PUBCONF NAMES H5pubconf.h PATHS ${HDF5_INCDIRS} NO_DEFAULT_PATH REQUIRED)
#	file(READ ${HDF5_PUBCONF} _hdf5_version_lines REGEX "#define[ \t]+H5_VERSION")
#	string(REGEX REPLACE ".*H5_VERSION .*\"\(.*\)\".*" "\\1" _hdf5_version "${_hdf5_version_lines}")
#	set(HDF5_VERSION "${_hdf5_version}" CACHE STRING "")
#	unset(_hdf5_version)
#	unset(_hdf5_version_lines)
#endif()
MESSAGE(STATUS "Found HDF5 libraries version ${HDF5_VERSION}")

#find_path(HDF5_INCLUDE_DIR "hdf5.h" ${HDF5_INCDIRS} NO_DEFAULT_PATH REQUIRED)
set(HDF5_INCLUDE_DIR ${HDF5_INCDIRS})
message(STATUS "Using HDF5 include dir: ${HDF5_INCLUDE_DIR}")

# NetCDF requires a pre-built HDF5 to work, so we can't package it in here :/
#CPMAddPackage(
#	NAME netCDF
#	GIT_REPOSITORY https://github.com/Unidata/netcdf-c
#	GIT_TAG        v4.9.0
#)

CPMAddPackage(
	NAME Libssh2
	GIT_REPOSITORY https://github.com/libssh2/libssh2
	GIT_TAG        libssh2-1.10.0
)

if(TARGET libssh2)
	add_library(Libssh2::libssh2 ALIAS libssh2)
endif()

if(FSC_WITH_PYTHON)
	CPMAddPackage(
		NAME pybind11
		GIT_REPOSITORY https://github.com/pybind/pybind11.git
		GIT_TAG        v2.9.1
	)
endif()

# For Botan, we have to be fancier, as it does not use CMake
CPMAddPackage(
	NAME Botan
	GIT_REPOSITORY https://github.com/randombit/botan.git
	GIT_TAG 2.19.0
	VERSION 2.19.0
	DOWNLOAD_ONLY
)

if(NOT TARGET Botan::botan)	
	include(src/cmake/CompileBotan.cmake)
endif()

# Re-enable BUILD_TESTING if desired
set(BUILD_TESTING ${BUILD_TESTING_TMP} CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES ${BUILD_EXAMPLES_TMP} CACHE BOOL "" FORCE)

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
	HDF5::hdf5-static
	Libssh2::libssh2
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
		DEPENDS fsc-python-bindings
    )
endif(SKBUILD)

# Install procedure

# Built in src/c++/fsc
INSTALL(TARGETS fsc cupnp deps EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(TARGET botan_selfbuilt)
	INSTALL(TARGETS botan_selfbuilt EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

	get_property(BOTAN_HEADERS TARGET botan_selfbuilt PROPERTY BUILT_HEADERS)
	INSTALL(DIRECTORY "${BOTAN_HEADERS}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

if(TARGET hdf5-static)
	target_include_directories(hdf5-static INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
	INSTALL(TARGETS hdf5-static EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
	INSTALL(DIRECTORY "${HDF5_SOURCE_DIR}/src/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
endif()

if(TARGET zlibstatic)
	target_include_directories(zlibstatic INTERFACE $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
	INSTALL(TARGETS zlibstatic EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
	INSTALL(DIRECTORY "${ZLIB_SOURCE_DIR}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
endif()

# FSC headers
# - Source headers
INSTALL(DIRECTORY src/c++/fsc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")

# - Genapi service headers
get_property(FSC_GENAPI_DIR TARGET fsc PROPERTY GENAPI_DIR)
INSTALL(DIRECTORY "${FSC_GENAPI_DIR}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")

if(FSC_WITH_PYTHON)
	if(SKBUILD)
		# If we build with scikit-build, this library needs to go into a special path that mirrors the package structure
		# Scikit will then automatically detect its presence and copy it into the source tree before packaging or creating
		# editable installs (!!!).
		INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION src/python/fsc COMPONENT SKBUILD)
	else()
		include(GNUInstallDirs)
		INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/site-packages/fsc)

		INSTALL(DIRECTORY src/python/ DESTINATION ${CMAKE_INSTALL_LIBDIR}/site-packages)
	endif()
endif()

INSTALL(EXPORT FSCTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake NAMESPACE FSC::)
