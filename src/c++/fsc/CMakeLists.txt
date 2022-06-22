# Set up capnpc options
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/genapi/fsc) # Set output dir

# Create output for .capnp files
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

set(
	FSC_CAPNP_SOURCES
	
	data.capnp
	data-test.capnp
	
	index.capnp
	
	magnetics.capnp
	magnetics-test.capnp
	
	geometry.capnp
	geometry-test.capnp
	
	http.capnp
	offline.capnp
	
	flt.capnp
	
	services.capnp
	
	devices/w7x.capnp
	devices/w7x-test.capnp
)
	

CAPNP_GENERATE_CPP(
	FSC_CP_SRC
	FSC_CP_HEADERS
	
	${FSC_CAPNP_SOURCES}
)

CUPNP_GEN(
	FSC_CP_CUHEADERS
	
	${FSC_CAPNP_SOURCES}
)

# Unset capnpc options
unset(CAPNPC_OUTPUT_DIR)
unset(CAPNPC_FLAGS)

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
	
	index.cpp
	
	geometry.h
	geometry.cpp
	
	magnetics.h
	magnetics-kernels.h
	magnetics.cpp
	
	flt.h
	flt-kernels.h
	flt.cpp
	
	kernels.h
	kernels.cpp
	
	http.h
	http.cpp
	
	devices/w7x.h
	devices/w7x.cpp
	
	grids.h
	tensor.h
	
	services.h
	services.cpp
	
	
	${FSC_CP_HEADERS}
	${FSC_CP_SRC}
	${FSC_CP_CUHEADERS}
)

set_property(
	TARGET fsc
	PROPERTY CP_HEADERS ${FSC_CP_HEADERS}
)

set_property(
	TARGET fsc
	PROPERTY CP_SRC ${FSC_CP_SRC}
)

set_property(
	TARGET fsc
	PROPERTY CP_CUHEADERS ${FSC_CP_CUHEADERS}
)

set_property(
	TARGET fsc
	PROPERTY SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(fsc PUBLIC deps cupnp)
target_include_directories(
	fsc
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/genapi>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	# TODO $<INSTALL_INTERFACE:???>
)

target_sources(
	tests
	PUBLIC
	local-tests.cpp
	random-tests.cpp
	data-tests.cpp
	geometry-tests.cpp
	magnetics-tests.cpp
	http-tests.cpp
	services-tests.cpp
	flt-tests.cpp
	devices/w7x-test.cpp
)
target_link_libraries(tests PUBLIC fsc)

if(FSC_WITH_CUDA)
	target_sources(
		fsc
		PRIVATE
		
		flt.cu
		magnetics.cu
	)
	set_target_properties(fsc PROPERTIES CUDA_ARCHITECTURES "60")
	set_target_properties(fsc PROPERTIES LINKER_LANGUAGE CUDA)
	set_target_properties(fsc PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_compile_options(fsc
		PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:
			-G
		>
	)
	target_link_options(fsc
		PRIVATE
		$<$<LINK_LANGUAGE:CUDA>:
			-G
		>
	)
	
	set_target_properties(tests PROPERTIES LINKER_LANGUAGE CUDA)
endif()