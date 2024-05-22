option(FSC_DEP_IGNORE_VERSIONS "Whether to ignore version requirements on dependencies" Off)
option(FSC_DEP_ALLOW_VENDORED "Whether to allow vendored dependencies" On)
option(FSC_DEP_PREF_VENDORED "Whether to preferentially use vendored dependencies" On)

macro(add_vendored_dependency)
	cmake_parse_arguments(AVDARGS "SETUP_ONLY" "PREFIX;NAME;VERSION" "FIND_ARGS" ${ARGN})
	
	option(FSC_DEP_${AVDARGS_NAME}_PREF_VENDORED "Whether to default to vendored dependency for ${AVDARGS_NAME}" Off)
			
	if(NOT SKBUILD AND (NOT FSC_DEP_PREF_VENDORED OR NOT FSC_DEP_ALLOW_VENDORED) AND NOT FSC_DEP_${AVDARGS_NAME}_PREF_VENDORED AND NOT (DEFINED FSC_DEP_${AVDARGS_NAME}_STRATEGY AND FSC_DEP_${AVDARGS_NAME}_STRATEGY STREQUAL "vendored"))
		if(${FSC_DEP_IGNORE_VERSIONS})
			message(STATUS "Searching for dependency ${AVDARGS_NAME} (any version)")
			find_package(${AVDARGS_NAME} ${AVDARGS_FIND_ARGS})
		else()
			message(STATUS "Searching for dependency ${AVDARGS_NAME} version ${AVDARGS_VERSION}")
			find_package(${AVDARGS_NAME} ${AVDARGS_VERSION} ${AVDARGS_FIND_ARGS})
		endif()
		
		if(${AVDARGS_NAME}_FOUND)
			message(STATUS "  Found")
			set(FSC_DEP_${AVDARGS_NAME}_STRATEGY "installed" CACHE STRING "Resolution strategy for dependency ${AVDARGS_NAME}")
		else()
			message(STATUS "  Not found")
			set(FSC_DEP_${AVDARGS_NAME}_STRATEGY "vendored" CACHE STRING "Resolution strategy for dependency ${AVDARGS_NAME}")
		endif()
	else()
		message(STATUS "Using vendored version of ${AVDARGS_NAME} due to previous choice or preference setting")
		set(FSC_DEP_${AVDARGS_NAME}_STRATEGY "vendored" CACHE STRING "Resolution strategy for dependency ${AVDARGS_NAME}")
	endif()

	if(FSC_DEP_${AVDARGS_NAME}_STRATEGY STREQUAL "vendored")	
		set(${AVDARGS_NAME}_VENDORED On)
		
		if(${FSC_DEP_ALLOW_VENDORED})
			message(STATUS "  Adding vendored version in ${AVDARGS_PREFIX}")
			
			if(NOT AVDARGS_SETUP_ONLY)
				add_subdirectory(${AVDARGS_PREFIX})
			endif()
						
			cmake_path(APPEND CMAKE_CURRENT_SOURCE_DIR "${AVDARGS_PREFIX}" OUTPUT_VARIABLE "${AVDARGS_NAME}_SOURCE_DIR")
			cmake_path(APPEND CMAKE_CURRENT_BINARY_DIR "${AVDARGS_PREFIX}" OUTPUT_VARIABLE "${AVDARGS_NAME}_BINARY_DIR")
			
			
			message(STATUS "  Src Dir: ${${AVDARGS_NAME}_SOURCE_DIR}")
			message(STATUS "  Bin Dir: ${${AVDARGS_NAME}_BINARY_DIR}")
		else()
			message(SEND_ERROR "${AVDARGS_NAME} not found, and vendored dependencies are disabled")
		endif()
	else()
		set(${AVDARGS_NAME}_VENDORED Off)
	endif()
endmacro()