find_package(Python3 REQUIRED COMPONENTS Interpreter)

message(STATUS "  Building botan from source")
if(MSVC)
	set(BOTAN_XARGS "--cc" "msvc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	set(BOTAN_XARGS "--cc" "gcc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	set(BOTAN_XARGS "--cc" "clang")
else()
	set(BOTAN_XARGS "")
endif()

#file(COPY ${Botan_SOURCE_DIR}/ DESTINATION ${Botan_BINARY_DIR})

set(AMALGAMATION_DIR "${Botan_BINARY_DIR}/amalgamation/$<CONFIG>")

add_custom_command(
	OUTPUT "${AMALGAMATION_DIR}/configure.py"
	
	COMMAND ${CMAKE_COMMAND} -E copy_directory ${Botan_SOURCE_DIR} ${AMALGAMATION_DIR}
)

add_custom_command(
	OUTPUT "${AMALGAMATION_DIR}/botan_all.cpp"
	DEPENDS "${AMALGAMATION_DIR}/configure.py"
	
	COMMAND ${Python3_EXECUTABLE}
	"${AMALGAMATION_DIR}/configure.py"
	"--minimized-build"
	"--enable-modules=auto_rng,blake2,sha2_32,sha2_64,system_rng"
	"--amalgamation"
	"--disable-shared"
	"--cc-bin" "${CMAKE_CXX_COMPILER}"
	"--with-build-dir" "${AMALGAMATION_DIR}"
	"--link-method" "copy"
	${BOTAN_XARGS}
	
	#COMMAND ${CMAKE_COMMAND} -E copy "${Botan_SOURCE_DIR}/botan_all.h" "${Botan_BINARY_DIR}/botan_all.h"
	#COMMAND ${CMAKE_COMMAND} -E copy "${Botan_SOURCE_DIR}/botan_all.cpp" "${Botan_BINARY_DIR}/botan_all.cpp"
	
	WORKING_DIRECTORY ${AMALGAMATION_DIR}
)

add_library(botan_selfbuilt "${AMALGAMATION_DIR}/botan_all.cpp")
target_include_directories(botan_selfbuilt INTERFACE "$<BUILD_INTERFACE:${AMALGAMATION_DIR}/build/include>")
target_include_directories(botan_selfbuilt INTERFACE "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
set_property(TARGET botan_selfbuilt PROPERTY BUILT_HEADERS "${AMALGAMATION_DIR}/build/include")


add_library(Botan::botan ALIAS botan_selfbuilt)
