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
	brandedEnums
	brandedEnums.cpp
)

target_link_libraries(brandedEnums CapnProto::capnp-rpc)

#add_executable(
#	npytest
#	npytest.cpp
#)
#
#target_link_libraries(npytest fsc pybind11::embed)
