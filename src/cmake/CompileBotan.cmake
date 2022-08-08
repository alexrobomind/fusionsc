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

# Note: FetchContent explicitly creates lower-case named variables
add_custom_command(
	OUTPUT "${botan_SOURCE_DIR}/botan_all.cpp"
	COMMAND ${Python3_EXECUTABLE}
	"${botan_SOURCE_DIR}/configure.py"
	"--minimized-build"
	"--enable-modules=auto_rng,blake2,sha2_32,sha2_64,system_rng"
	"--amalgamation"
	"--disable-shared"
	"--cc-bin" "${CMAKE_CXX_COMPILER}"
	${BOTAN_XARGS}
	WORKING_DIRECTORY ${botan_SOURCE_DIR}
)

add_library(botan_selfbuilt "${botan_SOURCE_DIR}/botan_all.cpp")
target_include_directories(botan_selfbuilt INTERFACE "${botan_SOURCE_DIR}/build/include")

add_library(Botan::botan ALIAS botan_selfbuilt)