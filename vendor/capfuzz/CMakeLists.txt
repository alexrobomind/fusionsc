add_library(
	capfuzz
	
	capfuzz.h
	capfuzz.cpp
)

target_link_libraries(
	capfuzz
	PUBLIC
	
	CapnProto::kj
	CapnProto::capnp-rpc
)

target_include_directories(
	capfuzz
	INTERFACE
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

install(TARGETS capfuzz EXPORT CapfuzzTargets)
INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")

install(EXPORT CapfuzzTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake NAMESPACE CapFuzz::)
export(EXPORT CapfuzzTargets NAMESPACE CapFuzz FILE CapfuzzExports.cmake)

add_library(CapFuzz::capfuzz ALIAS capfuzz)
	