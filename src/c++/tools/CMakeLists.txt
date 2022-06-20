add_executable(
	w7x-download-coils
	w7x-download-coils.cpp
)

target_link_libraries(w7x-download-coils fsc)

add_executable(
	fsc-server
	fsc-server.cpp
)

target_link_libraries(fsc-server fsc)