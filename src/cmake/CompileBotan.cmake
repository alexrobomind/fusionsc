find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "  Building botan from source")
if(MSVC)
	set(BOTAN_XARGS "--cc" "msvc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(BOTAN_XARGS "--cc" "gcc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(BOTAN_XARGS "--cc" "clang")
else()
	set(BOTAN_XARGS "")
endif()

# Note: FetchContent explicitly creates lower-case named variables
add_custom_command(
	OUTPUT "${Botan_SOURCE_DIR}/botan_all.cpp"
	COMMAND ${Python3_EXECUTABLE}
	"${Botan_SOURCE_DIR}/configure.py"
	"--minimized-build"
	"--enable-modules=auto_rng,blake2,sha2_32,sha2_64,system_rng"
	"--amalgamation"
	"--disable-shared"
	"--cc-bin" "${CMAKE_CXX_COMPILER}"
	${BOTAN_XARGS}
	WORKING_DIRECTORY ${Botan_SOURCE_DIR}
)

add_library(botan_selfbuilt "${Botan_SOURCE_DIR}/botan_all.cpp")
target_include_directories(botan_selfbuilt INTERFACE "$<BUILD_INTERFACE:${Botan_SOURCE_DIR}/build/include>")
target_include_directories(botan_selfbuilt INTERFACE "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET botan_selfbuilt PROPERTY BUILT_HEADERS "${Botan_SOURCE_DIR}/build/include")


add_library(Botan::botan ALIAS botan_selfbuilt)
