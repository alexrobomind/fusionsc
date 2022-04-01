if(FSC_WITH_PYTHON)
	pybind11_add_module(
		fscpy
		
		capnp.cpp
		dynamic_value.cpp
		fscpy.cpp
	)

	target_link_libraries(
		fscpy
		PUBLIC
		fsc
	)
endif()