find_library(BOTAN_LIB botan2)
find_path   (BOTAN_INC "botan/version.h")

if(BOTAN_LIB AND BOTAN_INC)
	set(Botan_FOUND TRUE)
	
	message("-- Found Botan")
	message("--     Library: ${BOTAN_LIB}")
	message("--     Include: ${BOTAN_INC}")
	
	add_library(Botan::botan INTERFACE)
	target_include_directories(Botan::botan INTERFACE ${BOTAN_INC})
	target_link_libraries(Botan::botan INTERFACE ${BOTAN_LIB})
else()
	message("-- Coult NOT find Botan")
	set(Botan_FOUND FALSE)
endif()