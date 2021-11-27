cmake_minimum_required(VERSION 3.14)

project(FSC LANGUAGES CXX)

# Dependencies:
# - capnproto (https://github.com/capnproto/capnproto)
# - Botan (https://github.com/randombit/botan)
# - Catch2 (https://github.com/catchorg/Catch2)

set(FSC_FETCH_DEPENDENCIES On CACHE BOOL "Whether the build system is allowed to fetch external dependencies")
set(FSC_ALWAYS_FETCH Off CACHE BOOL "Whether the build system is must fetch external dependencies")

# Prepare CTest
include(CTest)

# Try to load dependencies directly

if(NOT FSC_ALWAYS_FETCH)
	find_package(CapnProto)

	find_library(BOTAN_LIB botan2)
	find_path   (BOTAN_INC "botan/version.h")

	find_package(Catch2)
	
	if(BOTAN_LIB AND BOTAN_INC)
		set(Botan-Botan_FOUND TRUE)
	else()
		set(BotanBotan_FOUNDFOUND FALSE)
	endif()
else()
	set(CapnProto_FOUND FALSE)
	set(Botan_FOUND FALSE)
	set(Catch2_FOUND FALSE)
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
  GIT_REPOSITORY https://github.com/capnproto/capnproto.git
  GIT_TAG        v0.9.1
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
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    On
  GIT_PROGRESS   On
)

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

# For Botan, we have to be fancier, as it does not use CMake
if(Botan_FOUND)
	add_library(botan INTERFACE)
	target_include_directories(botan INTERFACE ${BOTAN_INC})
	target_link_libraries(botan INTERFACE ${BOTAN_LIB})
else()
	if(NOT FSC_FETCH_DEPENDENCIES)
		message(FATAL_ERROR "Botan could not be found, and dependency fetching was disabled")
	endif()

	find_package(Python3 REQUIRED COMPONENTS Interpreter)
	
	message(STATUS "Botan not found. Fetching and building from repository")
	message(STATUS "  Step 1: Download")
	FetchContent_MakeAvailable(Botan)
	
	message(STATUS "  Step 2: Amalgamation build")
	if(MSVC)
		set(BOTAN_XARGS "--cc" "msvc")
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		set(BOTAN_XARGS "--cc" "gcc")
	elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
		set(BOTAN_XARGS "--cc" "clang")
	else()
		set(BOTAN_XARGS "")
	endif()

	execute_process(
		COMMAND ${Python3_EXECUTABLE}
		"${botan_SOURCE_DIR}/configure.py"
		"--minimized-build"
		"--enable-modules=auto_rng,sha2_32,sha2_64,system_rng"
		"--amalgamation"
		"--disable-shared"
		"--cc-bin" "${CMAKE_CXX_COMPILER}"
		${BOTAN_XARGS}
		WORKING_DIRECTORY ${botan_SOURCE_DIR}
		COMMAND_ECHO STDOUT
	)
	
	add_library(botan "${botan_SOURCE_DIR}/botan_all.cpp")
	target_include_directories(botan INTERFACE "${botan_SOURCE_DIR}/build/include")
endif()

set(BUILD_TESTING ${BUILD_TESTING_TMP} CACHE BOOL "" FORCE)

add_library(deps INTERFACE)
target_link_libraries(
	deps
	INTERFACE
	botan
	CapnProto::capnp-rpc
	CapnProto::capnp-json
	CapnProto::kj-http
	Eigen3::Eigen
)

# Try to enable OpenMP
find_package(OpenMP 4.5)
if(OpenMP_FOUND)
	target_link_libraries(deps INTERFACE OpenMP::OpenMP_CXX)
endif()


# Set up testing subsystem
include(Catch)

add_subdirectory(src/c++)
