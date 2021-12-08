set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/fsc)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

CAPNP_GENERATE_CPP(
	FSC_CP_SRC
	FSC_CP_HEADERS
	
	data.capnp
	data-test.capnp
	
	magnetics.capnp
	magnetics-test.capnp
	
	geometry.capnp
	
	http.capnp
	offline.capnp
	
	devices/w7x.capnp
	devices/w7x-test.capnp
)
unset(CAPNPC_OUTPUT_DIR)

list(GET ${FSC_CP_HEADERS} 0 FSC_CP_HEADER1)

add_library(
	fsc_device
	
	device-gpulaunch.h
	device-grids.h
	device-kernels-biotsavart.h
	device-tensor.h
	
	device-dummy.cpp
)

if(FSC_WITH_CUDA)
	target_sources(
		fsc_device
		PRIVATE
		device-kernels.cu
	)
	set_target_properties(fsc_device PROPERTIES CUDA_ARCHITECTURES "60")
	set_target_properties(fsc_device PROPERTIES LINKER_LANGUAGE CUDA)
	set_target_properties(fsc_device PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
	target_compile_options(fsc_device
		PRIVATE
		$<$<COMPILE_LANGUAGE:CUDA>:
			-G
		>
	)
	target_link_options(fsc_device
		PRIVATE
		$<$<LINK_LANGUAGE:CUDA>:
			-G
		>
	)
endif()

target_link_libraries(fsc_device PUBLIC device_deps)

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
	
	geometry.h
	geometry.cpp
	
	magnetics.h
	magnetics-inl.h
	magnetics.cpp
	
	http.h
	http.cpp
	
	devices/w7x.h
	devices/w7x.cpp
	
	tensor.h
	
	${FSC_CP_HEADERS}
	${FSC_CP_SRC}
)

target_link_libraries(fsc PUBLIC deps fsc_device)
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
	magnetics-tests.cpp
	http-tests.cpp
	devices/w7x-test.cpp
)
target_link_libraries(tests PUBLIC fsc)

if(FSC_WITH_CUDA)
	set_target_properties(fsc PROPERTIES LINKER_LANGUAGE CUDA)
	#set_target_properties(fsc_device PROPERTIES LINKER_LANGUAGE CUDA)
	set_target_properties(tests PROPERTIES LINKER_LANGUAGE CUDA)
endif()