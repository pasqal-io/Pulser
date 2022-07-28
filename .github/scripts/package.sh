#!/usr/bin/env bash

# Exit if something fails
set -e

# Find and change to the repository directory
repo_dir=$(git rev-parse --show-toplevel)
cd "${repo_dir}"

# Removing existing files in /dist
rm -rf dist

if grep -q "dev" VERSION.txt; then
  echo "Development version"
  dev=true
else
  echo "Stable version"
  dev=false
fi

packages=$(cat packages.txt)
for pkg in $packages
do
  echo "Packaging $pkg"
  python $pkg/setup.py -q bdist_wheel -d "../dist"
  rm -r $pkg/build
done

if [ "$dev" = false ]; then
  python setup.py -q bdist_wheel -d "dist"
  rm -r build
fi

echo "Built wheels:"
ls dist
