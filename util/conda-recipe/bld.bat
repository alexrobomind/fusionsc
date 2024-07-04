mkdir build
cd build

cmake -E echo === Configuring ===
cmake -E echo Config command command cmake %CMAKE_ARGS% -DFSC_DEP_PREF_VENDORED=Off -DFSC_DEP_IGNORE_VERSIONS=On -DFSC_PYLIB_DIR:PATH=%SP_DIR% ..
cmake %CMAKE_ARGS% -DFSC_DEP_PREF_VENDORED=Off -DFSC_DEP_IGNORE_VERSIONS=On -DFSC_PYLIB_DIR:PATH=%SP_DIR% ..
if %errorlevel% neq 0 exit /b %errorlevel%

cmake -E echo === Building ===
cmake --build . --config Release 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%

cmake -E echo === Installing ===
cmake --install . --config Release 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%