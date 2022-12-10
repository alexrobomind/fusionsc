include(GNUInstallDirs)

add_library(fsc-vendor-sqlite3 sqlite3.c)
target_include_directories(fsc-vendor-sqlite3 INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}> $<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>)

install(TARGETS fsc-vendor-sqlite3 EXPORT FSCTargets LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES sqlite3.h sqlite3ext.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

add_library(SQLite::SQLite3 ALIAS fsc-vendor-sqlite3)