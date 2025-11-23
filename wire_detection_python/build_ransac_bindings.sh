#!/bin/bash
# Build script for ransac_bindings C++ extension module
# This script compiles ransac_bindings.cpp into a Python extension module (.so file)

set -e  # Exit on error

echo "Building ransac_bindings C++ extension module..."

# Check if required dependencies are installed
echo "Checking dependencies..."

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ is not installed. Install it with: sudo apt-get install build-essential"
    exit 1
fi

# Check for python3-config
if ! command -v python3-config &> /dev/null; then
    echo "Error: python3-dev is not installed. Install it with: sudo apt-get install python3-dev"
    exit 1
fi

# Check for pybind11
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo "Error: pybind11 is not installed. Install it with: pip install pybind11"
    exit 1
fi

# Check for Eigen3
if [ ! -d "/usr/include/eigen3" ] && [ ! -d "/usr/local/include/eigen3" ]; then
    echo "Warning: Eigen3 headers not found in standard locations."
    echo "Install it with: sudo apt-get install libeigen3-dev"
    EIGEN_INCLUDE=""
else
    if [ -d "/usr/include/eigen3" ]; then
        EIGEN_INCLUDE="-I/usr/include/eigen3"
    else
        EIGEN_INCLUDE="-I/usr/local/include/eigen3"
    fi
fi

# Check for OpenCV
if ! pkg-config --exists opencv4; then
    if ! pkg-config --exists opencv; then
        echo "Error: OpenCV is not installed or pkg-config cannot find it."
        echo "Install it with: sudo apt-get install libopencv-dev"
        exit 1
    else
        OPENCV_PKG="opencv"
    fi
else
    OPENCV_PKG="opencv4"
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build command
echo "Compiling ransac_bindings.cpp..."
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    $EIGEN_INCLUDE \
    $(pkg-config --cflags $OPENCV_PKG) \
    ransac_bindings.cpp \
    -o ransac_bindings$(python3-config --extension-suffix) \
    $(pkg-config --libs $OPENCV_PKG)

if [ $? -eq 0 ]; then
    echo "Success! ransac_bindings$(python3-config --extension-suffix) has been created."
    echo "You can now import it in Python with: import ransac_bindings"
else
    echo "Build failed!"
    exit 1
fi

