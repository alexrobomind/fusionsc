# Setup tests
add_executable(tests tests_main.cpp)
target_link_libraries(tests PUBLIC Catch2::Catch2WithMain deps)

set(SFC_TEST_DIR "${CMAKE_BINARY_DIR}/tests")

make_directory(${SFC_TEST_DIR})

catch_discover_tests(
	tests
	REPORTER junit
	OUTPUT_DIR ${SFC_TEST_DIR}
	OUTPUT_SUFFIX ".xml"
)

add_subdirectory(fsc)
add_subdirectory(tools)
