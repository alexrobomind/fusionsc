include(capnp_gen_java.cmake)

add_executable(capnpc-java capnpc-java.c++)

target_link_libraries(capnpc-java deps)