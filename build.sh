#!/bin/bash

# Build script for CNN library
# ANN and OpenCLWrapper are managed as Git submodules in extern/

SCRIPT_DIR="$(dirname "$0")"
cd "$SCRIPT_DIR"

# Initialize submodules if not already done
if [ ! -f "extern/ANN/CMakeLists.txt" ] || [ ! -f "extern/OpenCLWrapper/CMakeLists.txt" ]; then
    echo "Initializing Git submodules..."
    git submodule update --init --recursive
fi

# Update submodules to latest remote commit
echo "Updating submodules to latest..."
git submodule update --remote --merge

echo "Building CNN (with ANN and OpenCLWrapper)..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

