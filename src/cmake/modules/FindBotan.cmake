find_library(BOTAN_LIB botan-2 HINTS ENV SPACK_LINK_DIRS)
find_path   (BOTAN_INC "botan/version.h" HINTS ENV SPACK_INCLUDE_DIRS PATH_SUFFIXES botan-2)

message(STATUS "Boten search", ${BOTAN_LIB}, ${BOTAN_INC})

if(BOTAN_LIB AND BOTAN_INC)
	set(Botan_FOUND TRUE)
	
	message("-- Found Botan")
	message("--     Library: ${BOTAN_LIB}")
	message("--     Include: ${BOTAN_INC}")
	
	add_library(Botan::botan SHARED IMPORTED)
	set_property(TARGET Botan::botan PROPERTY IMPORTED_LOCATION ${BOTAN_LIB})
	target_include_directories(Botan::botan INTERFACE ${BOTAN_INC})

	# target_link_libraries(Botan::botan INTERFACE ${BOTAN_LIB})
else()
	message("-- Coult NOT find Botan")
	set(Botan_FOUND FALSE)
endif()
