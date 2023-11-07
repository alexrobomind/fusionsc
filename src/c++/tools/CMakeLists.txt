add_executable(
	fsc-tool
	
	fsc-tool.h
	fsc-tool.cpp
	
	fsc-tool-warehouse.cpp
	fsc-tool-worker.cpp
	fsc-tool-load-balancer.cpp
	fsc-tool-server.cpp
	fsc-tool-capnp.cpp
)	
target_link_libraries(fsc-tool fsc)
set_target_properties(fsc-tool PROPERTIES OUTPUT_NAME "fusionsc")

add_executable(
	fsc-fuzz
	EXCLUDE_FROM_ALL
	fuzz-test.cpp
)
target_link_libraries(fsc-fuzz fsc)


add_executable(
	brandedEnums
	EXCLUDE_FROM_ALL
	brandedEnums.cpp
)

target_link_libraries(brandedEnums CapnProto::capnp-rpc)

if(SKBUILD)
	add_dependencies(fsc-skbuild-target fsc-tool)
	install(TARGETS fsc-tool RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR} COMPONENT SKBUILD)
endif()

install(TARGETS fsc-tool EXPORT FSCTargets)
