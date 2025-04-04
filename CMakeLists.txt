cmake_minimum_required(VERSION 3.20)

# ===================== Project setup =====================

project(FSC LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/src/cmake/modules)

include(CTest)
include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build Shared Libraries" OFF)


# ===================== Language setup =====================

option(FSC_WITH_CUDA "Whether to use CUDA compilation")
if(FSC_WITH_CUDA)
	enable_language(CUDA)
	find_package(CUDAToolkit)
endif()

set(CMAKE_CXX_EXTENSIONS On)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(src/cmake/OptimizeForTarget.cmake)

# Enable auto-export of symbols to DLL
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ===================== Dependency setup =====================

include(src/cmake/AddVendoredDependency.cmake)

# Disable BUILD_TESTING while compiling dependencies
set(BUILD_TESTING_TMP ${BUILD_TESTING})
set(BUILD_TESTING OFF CACHE BOOL "Temporarily disable testing" FORCE)

set(BUILD_EXAMPLES_TMP ${BUILD_EXAMPLES})
set(BUILD_EXAMPLES OFF CACHE BOOL "Temporarily disable examples" FORCE)

# ----------- OpenMP -----------

find_package(OpenMP)

if(OPENMP_CXX_FOUND)
	set(FSC_OPENMP_DEFAULT On)
else()
	set(FSC_OPENMP_DEFAULT Off)
endif()

option(FSC_WITH_OPENMP "Whether to enable OpenMP support" ${FSC_OPENMP_DEFAULT})

if(FSC_WITH_OPENMP AND NOT OPENMP_CXX_FOUND)
	MESSAGE(SEND_ERROR "OpenMP requested, but not found")
endif()

# ----------- Python -----------

# Pre-load python to make sure we have compatible dev environment and interpreter
find_package(Python COMPONENTS Interpreter Development.Module NumPy)

message(STATUS "Python sitearch dir: ${Python_SITEARCH}")

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

# ----------- OpenSSl -----------

find_package(OpenSSL COMPONENTS Crypto SSL)

# ----------- Catch2 -----------

if(BUILD_TESTING_TMP)
	add_vendored_dependency(
		PREFIX vendor/catch2
		NAME Catch2
		VERSION 3.0
	)
endif()

# We need to include the test scan macro manually if we downloaded it manually
if(Catch2_VENDORED)
	list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
else()
	list(APPEND CMAKE_MODULE_PATH ${Catch2_DIR})
endif()
message(STATUS "Module path: ${CMAKE_MODULE_PATH}")

# ----------- ZLib -----------

set(SKIP_INSTALL_ALL On)
add_vendored_dependency(
	PREFIX vendor/zlib
	NAME ZLIB
	VERSION 1.2.0
)
unset(SKIP_INSTALL_ALL)

if(ZLIB_VENDORED)
	# Fix up includes for zlib
	set(ZLIB_CUSTOM_INCLUDES $<BUILD_INTERFACE:${ZLIB_SOURCE_DIR}> $<BUILD_INTERFACE:${ZLIB_BINARY_DIR}>)
	set_target_properties(zlib zlibstatic PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
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
	INSTALL(DIRECTORY "${ZLIB_BINARY_DIR}/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
else()
	add_library(ZLIB::zlib ALIAS ZLIB::ZLIB)
	add_library(ZLIB::zlibstatic ALIAS ZLIB::ZLIB)
endif()
set(ZLIB_FOUND 1)

set(ZLIB_STATIC_LIBRARY ZLIB::zlibstatic)
set(ZLIB_LIBRARY ZLIB::zlib)
set(ZLIB_INCLUDE_DIRECTORIES "")
set(ZLIB_INCLUDE_DIR "")

# ----------- ZLib -----------

add_vendored_dependency(
	PREFIX vendor/sqlite3
	NAME SQLite3
	VERSION 3.9.0
)

# ----------- Cap'n'proto -----------

# Bypass automatic ZLIB detection,
# as it doesn't work properly with
# our vendored ZLIB. Instead, we
# force it into the build below.
set(WITH_ZLIB OFF)
set(WITH_FIBERS ON CACHE STRING "Whether or not to build libkj-async with fibers")

# Currently, upstream Cap'n'proto
# has unfixed problems which prohibit
# us from using it properly.
# Until these are fixed, we need to
# use the vendored Cap'n'proto

#add_vendored_dependency(
# 	PREFIX vendor/capnproto
#	NAME CapnProto
#	VERSION 0.11
#)
add_subdirectory(vendor/capnproto)
set(CapnProto_VENDORED On)

unset(WITH_ZLIB)

# Enable ZLIB support for vendored Capnproto
if(CapnProto_VENDORED)
	target_compile_definitions(kj-http PRIVATE KJ_HAS_ZLIB)
	target_link_libraries(kj-http PUBLIC ZLIB::ZLIB)
	
	if(NOT TARGET CapnProto::capnpc)
		add_library(CapnProto::capnpc ALIAS capnpc)
	endif()
endif()

# ----------- yaml-cpp -----------

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

# ----------- Eigen3 -----------

set(EIGEN_BUILD_CMAKE_PACKAGE On)

add_vendored_dependency(
	PREFIX vendor/eigen3
	NAME Eigen3
	VERSION 3.4
)

# ----------- HDF5 -----------

set(FSC_WITH_HDF5 On)
set(HDF5_EXTERNALLY_CONFIGURED 1)
set(H5_ZLIB_HEADER "zlib.h")
set(HDF5_EXPORTED_TARGETS "FSCTargets")
set(LINK_COMP_LIBS ZLIB::ZLIB)

set(HDF5_BUILD_CPP_LIB ON CACHE BOOL "description" FORCE)

add_vendored_dependency(
	PREFIX vendor/hdf5
	NAME HDF5
	VERSION 1.12.0
	FIND_ARGS COMPONENTS HL C CXX
)

if(HDF5_VENDORED)
	set(HDF5_LIBS hdf5-static hdf5_hl-static hdf5_cpp-static hdf5_hl_cpp-static)
elseif(TARGET hdf5-shared)
	set(HDF5_LIBS hdf5-shared hdf5_hl-shared hdf5_cpp-shared hdf5_hl_cpp-shared)
else()
	set(HDF5_LIBS hdf5::hdf5 hdf5::hdf5_hl hdf5::hdf5_cpp hdf5::hdf5_hl_cpp)
endif()

# ----------- LibSSH2 -----------

add_vendored_dependency(
	PREFIX vendor/libssh2
	NAME Libssh2
	VERSION 1.10
)

if(Libssh2_VENDORED)
	target_compile_definitions(libssh2_static PRIVATE LIBSSH2DEBUG)
	set(LIBSSH2_LIBRARY libssh2_static)
elseif(TARGET Libssh2::libssh2)
	set(LIBSSH2_LIBRARY Libssh2::libssh2)
elseif(TARGET Libssh2::libssh2_static)
	set(LIBSSH2_LIBRARY Libssh2::libssh2_static)
elseif(TARGET Libssh2::libssh2_shared)
	set(LIBSSH2_LIBRARY Libssh2::libssh2_shared)
elseif(TARGET libssh2::libssh2)
	set(LIBSSH2_LIBRARY libssh2::libssh2)
elseif(TARGET libssh2::libssh2_static)
	set(LIBSSH2_LIBRARY libssh2::libssh2_static)
elseif(TARGET libssh2::libssh2_shared)
	set(LIBSSH2_LIBRARY libssh2::libssh2_shared)
else()
	MESSAGE(SEND_ERROR "Neither Libssh2::libssh2, Libssh2::libssh2_static, Libssh2::libssh2_shared, libssh2::libssh2, libssh2::libssh2_static, nor libssh2::libssh2_shared were found")
endif()

# ----------- Pybind11 -----------

if(FSC_WITH_PYTHON)
	add_vendored_dependency(
		PREFIX vendor/pybind11
		NAME pybind11
		VERSION 2.9.1
	)
endif()

# ----------- Botan -----------

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

# ----------- Jsonscons -----------

add_library(fusionsc_jsonscons INTERFACE)
target_include_directories(fusionsc_jsonscons INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/jsonscons/include>)
install(TARGETS fusionsc_jsonscons EXPORT FSCTargets)


# ----------- Happly -----------

add_subdirectory(vendor/happly)

# ----------- Poissongen -----------

add_library(fusionsc_poissongen INTERFACE)
target_include_directories(fusionsc_poissongen INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/vendor/pdg>)
install(TARGETS fusionsc_poissongen EXPORT FSCTargets)

# ----------- Capfuzz (internal) -----------

add_subdirectory(vendor/capfuzz)

# ===================== Dependency aggregation =====================

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
	${HDF5_LIBS}
	${LIBSSH2_LIBRARY}
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

# ===================== Build orchestration =====================

# Re-enable BUILD_TESTING if desired
set(BUILD_TESTING ${BUILD_TESTING_TMP} CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES ${BUILD_EXAMPLES_TMP} CACHE BOOL "" FORCE)

if(BUILD_TESTING)
	# Set up testing subsystem
	include(Catch)
endif()

# First build ancillary Cap'n'proto infrastructure
add_subdirectory(src/c++/cupnp) # Capnproto for CUDA
add_subdirectory(src/c++/capnpc-java) # Rudimentary Java compiler
add_subdirectory(src/c++/capnp-sphinx) # WIP Capnproto Sphinx bridge

# Compile the service interfaces
add_subdirectory(src/python/fusionsc/serviceDefs/fusionsc)

# Build and install the libraries & tools
add_subdirectory(src/c++)

# Build the documentation
add_subdirectory(src/docs)

# Install exports
install(EXPORT FSCTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake NAMESPACE FSC::)
export(EXPORT FSCTargets NAMESPACE fsc FILE FSCExports.cmake)

# ===================== Goodies =====================

# ----------- ASAN -----------

option(FSC_SANITIZE_ADDRESS "Whether to enable address sanitizer")

if(FSC_SANITIZE_ADDRESS)
	target_compile_options(deps INTERFACE "-fsanitize=address")
	target_link_options(deps INTERFACE "-fsanitize=address")
	
	if(CapnProto_VENDORED)
		target_compile_options(kj PUBLIC "-fsanitize=address")
		target_link_options(kj PUBLIC "-fsanitize=address")
	endif()
endif()

# ----------- march=native -----------

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

# ----------- CLANG coverage and instrumentation -----------

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

# ----------- Extra build targets -----------

# Generates a new Cap'n'proto id
add_custom_target(
	capnp-id
	COMMAND capnp_tool id
)

# Copies the library into the source folder
# (great for .pth dev installs)
if(FSC_WITH_PYTHON)
	add_custom_target(
		copy-pybindings
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:fsc-python-bindings> ${CMAKE_CURRENT_SOURCE_DIR}/src/python/fusionsc
	)
endif()

# ===================== Scikit build setup (pip build) =====================

if(SKBUILD)
	add_custom_target(fsc-skbuild-target)
	
	# FusionSC command line tool
	add_dependencies(fsc-skbuild-target fsc-tool)
	install(TARGETS fsc-tool RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR} COMPONENT SKBUILD)
	
	# FusionSC native python library
	add_dependencies(fsc-skbuild-target fsc-python-bindings)
	INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION ${SKBUILD_PLATLIB_DIR}/fusionsc COMPONENT SKBUILD)
endif()

# ===================== Summary =====================

message(STATUS "")
message(STATUS "--- Configuration options ---")
message(STATUS "Python support: ${FSC_WITH_PYTHON}")
message(STATUS "CUDA support: ${FSC_WITH_CUDA}")
message(STATUS "OpenMP support: ${OPENMP_CXX_FOUND}")
message(STATUS "OpenMP usage: ${FSC_WITH_OPENMP}")
message(STATUS "Native Arch support: ${FSC_ARCH_NATIVE_RESULT}")
message(STATUS "Tests: ${BUILD_TESTING}")
message(STATUS "Module path: ${CMAKE_MODULE_PATH}")
message(STATUS "")
message(STATUS "")
