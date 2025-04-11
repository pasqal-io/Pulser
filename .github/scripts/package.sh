#!/usr/bin/env bash

# Exit if something fails
set -e

# Find and change to the repository directory
repo_dir=$(git rev-parse --show-toplevel)
cd "${repo_dir}"

# Removing existing files in /dist
rm -rf dist

packages=$(cat packages.txt)
# Build the pulser packages
for pkg in $packages
do
  echo "Packaging $pkg"
  python -m build $pkg --wheel -o "../dist"
  rm -r $pkg/build
done

# Build the pulser metapackage
python -m build --wheel -o "dist"
rm -r build

echo "Built wheels:"
ls dist
