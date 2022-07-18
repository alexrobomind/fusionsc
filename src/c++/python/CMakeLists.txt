if(FSC_WITH_PYTHON)
	pybind11_add_module(
		native
		
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
		native
		PUBLIC
		fsc
	)
	
	if(SKBUILD)
		# If we build with scikit-build, this library needs to go into a special path that mirrors the package structure
		# Scikit will then automatically detect its presence and copy it into the source tree before packaging or creating
		# editable installs (!!!).
		INSTALL(TARGETS native LIBRARY DESTINATION src/python/fsc)
	else()
		INSTALL(TARGETS native LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
	endif()
endif()