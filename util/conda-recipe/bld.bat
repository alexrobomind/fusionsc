mkdir build
cd build

cmake -E echo === Configuring ===
cmake -E echo Config command command cmake %CMAKE_ARGS% -DFSC_DEP_PREF_VENDORED=Off -DFSC_DEP_IGNORE_VERSIONS=On -DFSC_PYLIB_DIR:PATH=%SP_DIR% ..
cmake %CMAKE_ARGS% -DFSC_DEP_PREF_VENDORED=Off -DFSC_DEP_IGNORE_VERSIONS=On -DFSC_PYLIB_DIR:PATH=%SP_DIR% ..

cmake -E echo === Building ===
cmake --build .

cmake -E echo === Installing ===
cmake --install .