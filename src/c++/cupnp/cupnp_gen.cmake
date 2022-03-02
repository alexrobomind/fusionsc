function(CUPNP_GEN HEADERS)
  if(NOT ARGN)
    message(SEND_ERROR "CUPNP_GEN() called without any source files.")
  endif()
  
  #Use cmake targets available
  if(TARGET capnp_tool)
    if(NOT CAPNP_EXECUTABLE)
      set(CAPNP_EXECUTABLE $<TARGET_FILE:capnp_tool>)
    endif()
    if(NOT CAPNP_INCLUDE_DIRECTORY)
      get_target_property(CAPNP_INCLUDE_DIRECTORY capnp_tool CAPNP_INCLUDE_DIRECTORY)
    endif()
    list(APPEND tool_depends capnp_tool)
  endif()
  if(NOT CAPNP_EXECUTABLE)
    message(SEND_ERROR "Could not locate capnp executable (CAPNP_EXECUTABLE).")
  endif()
  if(NOT CAPNP_INCLUDE_DIRECTORY)
    message(SEND_ERROR "Could not locate capnp header files (CAPNP_INCLUDE_DIRECTORY).")
  endif()

  if(DEFINED CAPNPC_OUTPUT_DIR)
    # Prepend a ':' to get the format for the '-o' flag right
    set(output_dir ":${CAPNPC_OUTPUT_DIR}")
  else()
    set(output_dir ":.")
  endif()

  if(NOT DEFINED CAPNPC_SRC_PREFIX)
    set(CAPNPC_SRC_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
  get_filename_component(CAPNPC_SRC_PREFIX "${CAPNPC_SRC_PREFIX}" ABSOLUTE)

  # Default compiler includes. Note that in capnp's own test usage of capnp_generate_cpp(), these
  # two variables will end up evaluating to the same directory. However, it's difficult to
  # deduplicate them because if CAPNP_INCLUDE_DIRECTORY came from the capnp_tool target property,
  # then it must be a generator expression in order to handle usages in both the build tree and the
  # install tree. This vastly overcomplicates duplication detection, so the duplication doesn't seem
  # worth fixing.
  set(include_path -I "${CAPNPC_SRC_PREFIX}" -I "${CAPNP_INCLUDE_DIRECTORY}")

  if(DEFINED CAPNPC_IMPORT_DIRS)
    # Append each directory as a series of '-I' flags in ${include_path}
    foreach(directory ${CAPNPC_IMPORT_DIRS})
      get_filename_component(absolute_path "${directory}" ABSOLUTE)
      list(APPEND include_path -I "${absolute_path}")
    endforeach()
  endif()

  set(${HEADERS})
  foreach(schema_file ${ARGN})
    get_filename_component(file_path "${schema_file}" ABSOLUTE)
    get_filename_component(file_dir "${file_path}" PATH)
    if(NOT EXISTS "${file_path}")
      message(FATAL_ERROR "Cap'n Proto schema file '${file_path}' does not exist!")
    endif()

    # Figure out where the output files will go
    if (NOT DEFINED CAPNPC_OUTPUT_DIR)
      set(CAPNPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/")
    endif()
    # Output files are placed in CAPNPC_OUTPUT_DIR, at a location as if they were
    # relative to CAPNPC_SRC_PREFIX.
    string(LENGTH "${CAPNPC_SRC_PREFIX}" prefix_len)
    string(SUBSTRING "${file_path}" 0 ${prefix_len} output_prefix)
    if(NOT "${CAPNPC_SRC_PREFIX}" STREQUAL "${output_prefix}")
      message(SEND_ERROR "Could not determine output path for '${schema_file}' ('${file_path}') with source prefix '${CAPNPC_SRC_PREFIX}' into '${CAPNPC_OUTPUT_DIR}'.")
    endif()

    string(SUBSTRING "${file_path}" ${prefix_len} -1 output_path)
    set(output_base "${CAPNPC_OUTPUT_DIR}${output_path}")

    add_custom_command(
      OUTPUT "${output_base}.cu.h"
      COMMAND "${CAPNP_EXECUTABLE}"
      ARGS compile
          -o $<TARGET_FILE:cupnpc>${output_dir}
          --src-prefix ${CAPNPC_SRC_PREFIX}
          ${include_path}
          ${CAPNPC_FLAGS}
          ${file_path}
      DEPENDS "${schema_file}" ${tool_depends} cupnpc
      COMMENT "Compiling Cap'n Proto schema ${schema_file} (CuPnP)"
      VERBATIM
    )

    list(APPEND ${HEADERS} "${output_base}.cu.h")
  endforeach()

  set_source_files_properties(${${SOURCES}} ${${HEADERS}} PROPERTIES GENERATED TRUE)
  set(${HEADERS} ${${HEADERS}} PARENT_SCOPE)
endfunction()