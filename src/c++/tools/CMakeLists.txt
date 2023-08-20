add_executable(
	fsc-server
	fsc-server.cpp
)

target_link_libraries(fsc-server fsc)

add_executable(
	fsc-load-balancer
	load-balancer.cpp
)
target_link_libraries(fsc-load-balancer fsc)

add_executable(
	fsc-worker
	worker.cpp
)
target_link_libraries(fsc-worker fsc)

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

install(TARGETS fsc-server fsc-load-balancer EXPORT FSCTargets)
