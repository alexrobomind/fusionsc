:: Builds the documentation and pushes it to the main repository
:: Builds from 'docs' to 'docs' (because of github pages limitations)
:: The Sphinx "docs" CMake target now includes C++ docs via Breathe
:: (when Doxygen is available)

:: Switch to 'pages' without checkout
git symbolic-ref HEAD refs/heads/pages

:: Overwrite branch with contents of main
git reset --hard main

:: Build Sphinx documentation (includes C++ via Breathe when Doxygen is found)
rmdir /s /q docs
cd build
cmake --build . --target copy-docs --config Release
cd ..

:: Disable the Jekyll preprocessor
echo "No Jekyll" > docs\.nojekyll

:: Commit & force push
git add -A docs
git commit -m "Documentation update"
git push -f origin pages

:: Go back to main
git checkout main
