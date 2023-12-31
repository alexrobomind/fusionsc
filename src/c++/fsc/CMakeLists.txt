add_library(
	fsc
	
	common.h
	common.cpp
	
	typing.h
	typing.cpp
	
	local.h
	local.cpp
	
	store.h
	store.cpp
	
	matcher.h
	matcher.cpp
	
	ssh.h
	ssh.cpp
	
	streams.h
	streams.cpp
	
	data.h
	data.cpp
	
	yaml.h
	yaml.cpp
	
	textio.h
	textio.cpp
	textio-node.cpp
	textio-json.cpp
	textio-yaml.cpp
	
	hdf5.h
	hdf5.cpp
	
	networking.h
	networking.cpp
	
	compression.h
	compression.cpp
	
	jobs.h
	jobs.cpp
	jobs-unix.cpp
	jobs-win32.cpp
	jobs-slurm.cpp
	jobs-mpi.cpp
	
	db.h
	db.cpp
	
	sqlite.h
	sqlite.cpp
	
	blob-store.h
	blob-store.cpp
	
	db-cache.h
	db-cache.cpp
	
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
	
	offline.h
	offline.cpp
	
	flt.h
	flt-kernels.h
	flt.cpp
	
	hfcam.h
	hfcam.cpp
	
	poly.h
	poly.cpp
	
	http.h
	http.cpp
	
	devices/w7x.h
	devices/w7x.cpp
	
	devices/jtext.h
	devices/jtext.cpp
	
	grids.h
	tensor.h
	
	services.h
	services.cpp
	
	random.h
	
	interpolation.h
	
	fieldline-mapping.h
	fieldline-mapping.cpp
	
	load-balancer.h
	load-balancer.cpp
	
	kernels/array.h
	kernels/device.h
	kernels/karg.h
	kernels/kernels.h
	kernels/launch.h
	kernels/launch-inl.h
	kernels/message.h
	kernels/tensor.h
	
	kernels/device.cpp
	kernels/device-dummy.cpp
	kernels/message.cpp
)

target_link_libraries(fsc PUBLIC deps cupnp fsc-service-genapi)
target_link_libraries(fsc PRIVATE fusionsc_jsonscons)

target_include_directories(
	fsc
	PUBLIC
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(BUILD_TESTING)
	target_sources(
		tests
		PUBLIC
		poly-tests.cpp
		local-tests.cpp
		matcher-tests.cpp
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
		vmec-tests.cpp
		blob-store-tests.cpp
		db-cache-tests.cpp
		textio-tests.cpp
		streams-tests.cpp
		
		efit-tests.cpp
		
		devices/w7x-test.cpp
		devices/jtext-test.cpp
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

# Built in src/c++/fsc
INSTALL(TARGETS deps EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# FSC headers
# - Source headers
INSTALL(DIRECTORY src/c++/fsc/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")
