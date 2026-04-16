#!/usr/bin/env bash
set -euo pipefail

# Builds the documentation and pushes it to the GitHub Pages branch
# The Sphinx "docs" CMake target now includes C++ docs via Breathe
# (when Doxygen is available)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if [ ! -d "${BUILD_DIR}" ]; then
  echo "Error: Build directory not found at ${BUILD_DIR}"
  echo "Run cmake configure first:"
  echo "  mkdir build && cd build && cmake .."
  exit 1
fi

# Switch to 'pages' branch without checkout
git symbolic-ref HEAD refs/heads/pages
git reset --hard main

# Clean old docs
rm -rf docs

# Build Sphinx documentation (includes C++ via Breathe when Doxygen is found)
cmake --build "${BUILD_DIR}" --target copy-docs --config Release

# Disable the Jekyll preprocessor
echo "No Jekyll" > docs/.nojekyll

# Commit & force push
git add -A docs
git commit -m "Documentation update"
git push -f origin pages

# Go back to main
git checkout main
