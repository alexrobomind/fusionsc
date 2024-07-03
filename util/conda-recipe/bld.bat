mkdir build
cd build
cmake -DFSC_DEP_PREF_VENDORED=Off -DFSC_DEP_IGNORE_VERSIONS=On -DBUILD_SHARED_LIBS=On ..
cmake --build . --target fsc-python-bindings fsc-tool --config Release
cmake --install . --config Release