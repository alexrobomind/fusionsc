# Set up capnpc options
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/genapi/fsc) # Set output dir

# Create output for .capnp files
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

set(
	FSC_CAPNP_SOURCES
	
	data.capnp
	data-archive.capnp
	streams.capnp
	
	index.capnp
	odb.capnp
	jobs.capnp
	
	magnetics.capnp
	geometry.capnp
	
	vmec.capnp
	
	http.capnp
	offline.capnp
	
	random.capnp
	flt.capnp
	hfcam.capnp
	
	local-vat-network.capnp
	
	networking.capnp
	services.capnp
	
	java.capnp
	
	devices/w7x.capnp
	devices/w7x-geometry.capnp
)

set(
	FSC_CAPNP_TEST_SOURCES
	
	data-test.capnp
	magnetics-test.capnp
	geometry-test.capnp
	devices/w7x-test.capnp
)
	
	

CAPNP_GENERATE_CPP(
	FSC_CP_SRC
	FSC_CP_HEADERS
	
	${FSC_CAPNP_SOURCES}
	${FSC_CAPNP_TEST_SOURCES}
)

CUPNP_GEN(
	FSC_CP_CUHEADERS
	
	${FSC_CAPNP_SOURCES}
	${FSC_CAPNP_TEST_SOURCES}
)

set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java-classes/org/fsc) # Set output dir
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

CAPNP_GEN_JAVA(
	java-classes
	
	${FSC_CAPNP_SOURCES}
)

set_property(TARGET java-classes PROPERTY FSC_JAVA_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/java-classes)
set_property(TARGET java-classes APPEND PROPERTY EXPORT_PROPERTIES FSC_JAVA_SOURCES)
	
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
	
	ssh.h
	ssh.cpp
	
	streams.h
	streams.cpp
	
	data.h
	data.cpp
	
	yaml.h
	yaml.cpp
	
	networking.h
	networking.cpp
	
	cudata.h
	cudata.cpp
	
	compression.h
	compression.cpp
	
	jobs.h
	jobs.cpp
	jobs-unix.cpp
	jobs-win32.cpp
	jobs-slurm.cpp
	
	db.h
	db.cpp
	
	odb.h
	odb.cpp
	
	local-vat-network.h
	local-vat-network.cpp
	
	index.h
	index.cpp
	
	vmec.h
	vmec.cpp
	
	efit.h
	efit.cpp
	
	geometry.h
	geometry.cpp
	
	magnetics.h
	magnetics-kernels.h
	magnetics.cpp
	
	device.h
	device.cpp
	
	flt.h
	flt-kernels.h
	flt.cpp
	
	kernels.h
	kernels.cpp
	
	hfcam.h
	hfcam.cpp
	
	http.h
	http.cpp
	
	devices/w7x.h
	devices/w7x.cpp
	
	grids.h
	tensor.h
	
	services.h
	services.cpp
	
	random.h
	
	interpolation.h
	
	fieldline-mapping.h
	fieldline-mapping.cpp
	
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
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_property(TARGET fsc PROPERTY GENAPI_DIR "${CMAKE_CURRENT_BINARY_DIR}/genapi")

if(BUILD_TESTING)
	target_sources(
		tests
		PUBLIC
		local-tests.cpp
		jobs-tests.cpp
		random-tests.cpp
		data-tests.cpp
		odb-tests.cpp
		geometry-tests.cpp
		magnetics-tests.cpp
		http-tests.cpp
		services-tests.cpp
		flt-tests.cpp
		interpolation-tests.cpp
		fieldline-mapping-tests.cpp
		
		efit-tests.cpp
		
		devices/w7x-test.cpp
	)
	target_link_libraries(tests PUBLIC fsc)
endif()

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
