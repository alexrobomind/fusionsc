if(FSC_WITH_PYTHON)
	pybind11_add_module(
		pybindfsc
		
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
		pybindfsc
		PUBLIC
		fsc
	)
	
	INSTALL(TARGETS pybindfsc LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/site-packages/fsc)
endif()