:: Builds the documentation and pushes it to the main repository
:: Builds from 'docs' to 'docs' (because of github pages limitations)

:: Checkout
git checkout pages
git reset --hard main

:: Build documentations into "docs" dir in src tree
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
