add_library(happly INTERFACE)

target_include_directories(
	happly INTERFACE
	
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
	$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_library(happly::happly ALIAS happly)

install(TARGETS happly EXPORT FSCTargets)
install(FILES happly.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})