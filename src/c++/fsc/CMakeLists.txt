set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/fsc)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

CAPNP_GENERATE_CPP(
	FSC_CP_SRC
	FSC_CP_HEADERS
	
	data.capnp
	data-test.capnp
	magnetics.capnp
	http.capnp
	offline.capnp
	
	devices/w7x.capnp
	devices/w7x-test.capnp
)
unset(CAPNPC_OUTPUT_DIR)

list(GET ${FSC_CP_HEADERS} 0 FSC_CP_HEADER1)


add_library(
	fsc
	
	common.h
	common.cpp
	
	local.h
	local.cpp
	
	store.h
	store.cpp
	
	data.h
	data.cpp
	
	magnetics.h
	magnetics.cpp
	magnetics-fieldcalc.cpp
	
	http.h
	http.cpp
	
	devices/w7x.h
	devices/w7x.cpp
	
	tensor.h
	
	${FSC_CP_HEADERS}
	${FSC_CP_SRC}
)
target_link_libraries(fsc PUBLIC deps)
target_include_directories(
	fsc
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	# TODO $<INSTALL_INTERFACE:???>
)

target_sources(
	tests
	PUBLIC
	local-tests.cpp
	random-tests.cpp
	data-tests.cpp
	http-tests.cpp
	devices/w7x-test.cpp
)
target_link_libraries(tests PUBLIC fsc)