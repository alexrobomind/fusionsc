if(FSC_WITH_PYTHON)
	pybind11_add_module(
		fsc-python-bindings
		
		assign.cpp
		graphviz.cpp
		tensor.cpp		
		
		capnp-impl.cpp
		capnp-bind.cpp
		pickle.cpp
		
		structio.cpp
		
		kj.cpp
		fscpy.cpp
		async.cpp
		loader.cpp
		loader-parse.cpp
		loader-make-types.cpp
		loader-interface-method.cpp
		data.cpp
		devices.cpp
		serialize.cpp
		
		service.cpp
		
		helpers.cpp
	)

	target_link_libraries(
		fsc-python-bindings
		PRIVATE
		fsc
	)

	set_target_properties(fsc-python-bindings PROPERTIES OUTPUT_NAME "native")
	
	if(DEFINED FSC_PYLIB_DIR)
		set(INSTALL_DIR ${FSC_PYLIB_DIR})
	else()
		set(INSTALL_DIR ${Python_SITEARCH})
	endif()
	
	message(STATUS "Python package installation root: ${INSTALL_DIR}")
	
	INSTALL(TARGETS fsc-python-bindings LIBRARY DESTINATION ${INSTALL_DIR}/fusionsc)
	INSTALL(DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/fusionsc/ DESTINATION ${INSTALL_DIR}/fusionsc FILES_MATCHING PATTERN "*.py" PATTERN "*.capnp")
endif()
