# Look for documentation generator
find_package(Doxygen)

get_target_property(FSC_CP_HEADERS fsc CP_HEADERS)
get_target_property(FSC_CP_CUHEADERS fsc CP_CUHEADERS)
get_target_property(FSC_CP_SRC fsc CP_SRC)
get_target_property(FSC_SRC_DIR fsc SRC_DIR)

list(APPEND GENAPI_HEADERS ${FSC_CP_HEADERS})
# list(APPEND GENAPI_HEADERS ${FSC_CP_CUHEADERS})
list(JOIN GENAPI_HEADERS " " FSC_GENAPI)

list(REMOVE_ITEM FSC_SRC ${FSC_CP_HEADERS} ${FSC_CP_SRC} ${FSC_CP_CUHEADERS})

set(FSC_RAWSRC "${FSC_SRC_DIR}")

# Configure documentation files
configure_file(gendocs-doxyfile.in gendocs-doxyfile)
configure_file(fsc-doxyfile.in fsc-doxyfile)

# List dependencies of main documentation
file(GLOB_RECURSE FSC_DOCSOURCES "${FSC_SRC_DIR}/*")

add_custom_command(
	OUTPUT genapi.doxytag
	DEPENDS ${FSC_CP_HEADERS} ${FSC_CH_CUHEADERS} Doxygen::doxygen gendocs-doxyfile fsc
	COMMAND Doxygen::doxygen
	ARGS gendocs-doxyfile
)

add_custom_command(
	OUTPUT main.doxytag
	DEPENDS ${FSC_DOCSOURCES} Doxygen::doxygen fsc-doxyfile
	COMMAND Doxygen::doxygen
	ARGS fsc-doxyfile
)

add_custom_target(
	genapi-docs
	DEPENDS genapi.doxytag
)

add_custom_target(
	doxy-docs
	DEPENDS main.doxytag
)

if(FSC_WITH_PYTHON)
	set(DOC_FSCPATH ${CMAKE_CURRENT_BINARY_DIR}/docfsc)
	file(MAKE_DIRECTORY ${DOC_FSCPATH}/fsc)

	set(DOC_SPHINXPATH ${CMAKE_CURRENT_BINARY_DIR}/sphinx_docs)
	set(DOC_TOCTREE_PATH ${CMAKE_CURRENT_BINARY_DIR}/sphinx_toctrees)

	add_custom_target(
		docs-setup1
		COMMAND ${CMAKE_COMMAND} -E copy_directory ${PROJECT_SOURCE_DIR}/src/python ${DOC_FSCPATH}
	)

	add_custom_target(
		docs-setup2
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:fsc-python-bindings> ${DOC_FSCPATH}/fsc
	)

	set(DOC_JAVADOCS ${CMAKE_CURRENT_SOURCE_DIR}/javadocs)
	file(MAKE_DIRECTORY ${DOC_JAVADOCS})

	add_custom_target(
		docs-prepare-java
		COMMAND Python::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/javadoc.py $<TARGET_PROPERTY:java-classes,FSC_JAVA_SOURCES> ${DOC_JAVADOCS}
		DEPENDS java-classes
	)

	add_custom_target(
		docs
		COMMAND ${CMAKE_COMMAND} -E env DOC_FSCPATH=${DOC_FSCPATH} sphinx-build -a -d ${DOC_TOCTREE_PATH} ${CMAKE_CURRENT_SOURCE_DIR} ${DOC_SPHINXPATH}
	)

	add_dependencies(docs docs-setup1 docs-setup2)# docs-prepare-java
endif()
