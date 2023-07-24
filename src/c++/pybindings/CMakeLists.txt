if(FSC_WITH_PYTHON)
	pybind11_add_module(
		fsc-python-bindings
		
		assign.cpp
		graphviz.cpp
		tensor.cpp		
		
		capnp-impl.cpp
		capnp-bind.cpp
		
		kj.cpp
		fscpy.cpp
		async.cpp
		loader.cpp
		data.cpp
		devices.cpp
		
		service.cpp
		
		helpers.cpp
	)
	# capnp.cpp

	target_link_libraries(
		fsc-python-bindings
		PUBLIC
		fsc
	)

	set_target_properties(fsc-python-bindings PROPERTIES OUTPUT_NAME "native")
	
	if(SKBUILD)
		# If we build with scikit-build, this library needs to go into a special path that mirrors the package structure
		# Scikit will then automatically detect its presence and copy it into the source tree before packaging or creating
		# editable installs (!!!).
		INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION src/python/fusionsc COMPONENT SKBUILD)
		
		add_custom_target(fsc-install-skbuild
			${CMAKE_COMMAND}
			-DCMAKE_INSTALL_COMPONENT=SKBUILD
			-P "${PROJECT_BINARY_DIR}/cmake_install.cmake"
			DEPENDS fsc-python-bindings
		)
	else()		
		if(DEFINED FSC_PYLIB_DIR)
			set(INSTALL_DIR ${FSC_PYLIB_DIR})
		else()
			set(INSTALL_DIR ${Python_SITEARCH})
		endif()
		
		message(STATUS "Python package installation root: ${INSTALL_DIR}")
		
		INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION ${INSTALL_DIR}/fusionsc)
		INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/fusionsc/ DESTINATION ${INSTALL_DIR}/fusionsc FILES_MATCHING PATTERN "*.py")
	endif()
endif()
