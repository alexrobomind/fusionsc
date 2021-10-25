# Setup tests
add_executable(tests tests_main.cpp)
target_link_libraries(tests Catch2::Catch2WithMain deps)

catch_discover_tests(tests)