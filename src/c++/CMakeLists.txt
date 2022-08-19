# Setup tests
if(BUILD_TESTING)
	add_executable(tests tests_main.cpp)
	target_link_libraries(tests PUBLIC Catch2::Catch2 deps)

	set(SFC_TEST_DIR "${CMAKE_BINARY_DIR}/tests")

	make_directory(${SFC_TEST_DIR})

	catch_discover_tests(
		tests
		REPORTER junit
		OUTPUT_DIR ${SFC_TEST_DIR}
		OUTPUT_SUFFIX ".xml"
	)
endif()

# First build the cupnp library / compiler
add_subdirectory(cupnp)

# Then build the fsc library
add_subdirectory(fsc)

# Now we can build tools / bindings
add_subdirectory(tools)
add_subdirectory(pybindings)
