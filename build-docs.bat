:: Builds the documentation and pushes it to the main repository
:: Builds from 'docs' to 'docs' (because of github pages limitations)

git checkout pages
git reset --hard main

rmdir /s /q docs

cd build
cmake --build . --target copy-docs --config Release
cd ..

git add -A docs
git commit -m "Documentation update"
git push -f origin pages

:: Go back to main and delete documentation
git checkout main

rmdir /s /q docs

