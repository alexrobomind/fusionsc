# Set up capnpc options
set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/genapi/fsc) # Set output dir

# Create output for .capnp files
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

set(
	FSC_CAPNP_SOURCES
	
	data.capnp
	data-archive.capnp
	streams.capnp
	
	warehouse.capnp
	warehouse-internal.capnp
	
	index.capnp
	jobs.capnp
	matcher.capnp
	
	magnetics.capnp
	geometry.capnp
	
	vmec.capnp
	hint.capnp
	
	http.capnp
	offline.capnp
	
	random.capnp
	flt.capnp
	hfcam.capnp
	
	local-vat-network.capnp
	
	networking.capnp
	services.capnp
	
	java.capnp
	
	devices/w7x.capnp	
	devices/jtext.capnp
	
	dynamic.capnp
)

set(
	FSC_CAPNP_TEST_SOURCES
	
	data-test.capnp
	magnetics-test.capnp
	geometry-test.capnp
	devices/w7x-test.capnp
)
	
	

CAPNP_GENERATE_CPP(
	FSC_CP_SRC
	FSC_CP_HEADERS
	
	${FSC_CAPNP_SOURCES}
	${FSC_CAPNP_TEST_SOURCES}
)

CUPNP_GEN(
	FSC_CP_CUHEADERS
	
	${FSC_CAPNP_SOURCES}
	${FSC_CAPNP_TEST_SOURCES}
)

set(CAPNPC_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/java-classes/org/fsc) # Set output dir
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})

CAPNP_GEN_JAVA(
	java-classes
	
	${FSC_CAPNP_SOURCES}
)

set_property(TARGET java-classes PROPERTY FSC_JAVA_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/java-classes)
set_property(TARGET java-classes APPEND PROPERTY EXPORT_PROPERTIES FSC_JAVA_SOURCES)
	
# Unset capnpc options
unset(CAPNPC_OUTPUT_DIR)
unset(CAPNPC_FLAGS)

add_library(
	fsc-service-genapi
	
	${FSC_CP_HEADERS}
	${FSC_CP_SRC}
	${FSC_CP_CUHEADERS}
)

target_link_libraries(fsc-service-genapi PUBLIC deps cupnp)
target_include_directories(fsc-service-genapi PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/genapi>)
target_include_directories(fsc-service-genapi PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS fsc-service-genapi EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/genapi/" DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")

set_property(TARGET fsc-service-genapi	PROPERTY CP_HEADERS ${FSC_CP_HEADERS})
set_property(TARGET fsc-service-genapi	PROPERTY CP_SRC ${FSC_CP_SRC})
set_property(TARGET fsc-service-genapi PROPERTY CP_CUHEADERS ${FSC_CP_CUHEADERS})
set_property(TARGET fsc-service-genapi PROPERTY SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET fsc-service-genapi PROPERTY GENAPI_DIR "${CMAKE_CURRENT_BINARY_DIR}/genapi")