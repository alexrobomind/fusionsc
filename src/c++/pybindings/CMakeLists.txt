if(FSC_WITH_PYTHON)
	pybind11_add_module(
		fsc-python-bindings
		
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
		fsc-python-bindings
		PUBLIC
		fsc
	)

	set_target_properties(fsc-python-bindings PROPERTIES OUTPUT_NAME "native")
endif()
