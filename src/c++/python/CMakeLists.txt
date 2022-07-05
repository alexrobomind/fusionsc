if(FSC_WITH_PYTHON)
	pybind11_add_module(
		fscpy
		
		capnp.cpp
		kj.cpp
		fscpy.cpp
		async.cpp
		loader.cpp
		data.cpp
		devices.cpp
		
		service.cpp
	)

	target_link_libraries(
		fscpy
		PUBLIC
		fsc
	)
endif()