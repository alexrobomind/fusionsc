if(FSC_WITH_PYTHON)
	add_library(
		fscpy
		
		capnp.cpp
	)

	target_link_libraries(
		fscpy
		PUBLIC
		fsc
	)
endif()