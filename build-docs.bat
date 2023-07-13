:: Builds the documentation and pushes it to the main repository
:: Builds from 'docs' to 'docs' (because of github pages limitations)

git checkout pages
rmdir /s /q docs
git reset --hard main
cd build
cmake --build . --target docs --config Release
cd ..
rmdir /s /q docs
mkdir docs
xcopy /f /s build\docs\sphinx_docs docs
git add -A docs
git commit -m "Documentation update"
git push -f origin pages

:: Go back to main
git checkout main
