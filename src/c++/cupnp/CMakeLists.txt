include(cupnp_gen.cmake)

add_executable(cupnpc cupnpc.cpp)

target_link_libraries(cupnpc deps)

add_library(cupnp INTERFACE)
target_include_directories(
	cupnp
	INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	# TOD: INSTALL_INTERFACE
)

target_compile_features(cupnp INTERFACE cxx_std_17)

set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

CAPNP_GENERATE_CPP(CUPNP_TEST_HEADERS CUPNP_TEST_SOURCES cupnp-test.capnp)
CUPNP_GEN(CUPNP_TEST_CUHEADERS cupnp-test.capnp)

add_executable(cupnp_test test.cpp ${CUPNP_TEST_HEADERS} ${CUPNP_TEST_HEADERS} ${CUPNP_TEST_CUHEADERS})
target_include_directories(cupnp_test PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(cupnp_test Catch2::Catch2WithMain CapnProto::capnp CapnProto::kj cupnp)