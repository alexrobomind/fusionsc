# Setup tests
if(BUILD_TESTING)
	add_executable(tests EXCLUDE_FROM_ALL tests_main.cpp)
	target_link_libraries(tests PUBLIC Catch2::Catch2 deps)

	set(SFC_TEST_DIR "${CMAKE_BINARY_DIR}/tests")

	make_directory(${SFC_TEST_DIR})

	catch_discover_tests(
		tests
		REPORTER junit
		OUTPUT_DIR ${SFC_TEST_DIR}
		OUTPUT_SUFFIX ".xml"
	)
	
	add_custom_target(
		run-tests
		COMMAND ctest .
		DEPENDS tests
	)
	
	# If we build ZLIB ourselves, we need to make sure that its test targets are also built
	if(ZLIB_SOURCE_DIR)
		# These two targets come from ZLIB (bleh, we're lucky there are no conflicts)
		if(TARGET example)
			add_dependencies(tests example)
		endif()
		if(TARGET example64)
			add_dependencies(tests example64)
		endif()
	endif()
endif()

# Cupnp is included from the root

# Then build the fsc library
add_subdirectory(fsc)

# Now we can build tools / bindings
add_subdirectory(tools)
add_subdirectory(pybindings)

# Some fun extra targets
add_custom_target(
	capnp-id
	COMMAND capnp_tool id
)

if(FSC_WITH_PYTHON)
	add_custom_target(
		copy-pybindings
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:fsc-python-bindings> ${CMAKE_CURRENT_SOURCE_DIR}/../python/fusionsc
	)
endif()