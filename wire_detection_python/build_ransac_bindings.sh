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
OPENCV_PKG=""
OPENCV_CFLAGS=""
OPENCV_LIBS=""

# Allow manual override via environment variables (useful for Jetson)
if [ -n "$OPENCV_INCLUDE_DIR" ] && [ -n "$OPENCV_LIB_DIR" ]; then
    OPENCV_CFLAGS="-I$OPENCV_INCLUDE_DIR"
    OPENCV_LIBS="-L$OPENCV_LIB_DIR -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
    echo "Using OpenCV from environment variables:"
    echo "  OPENCV_INCLUDE_DIR=$OPENCV_INCLUDE_DIR"
    echo "  OPENCV_LIB_DIR=$OPENCV_LIB_DIR"
elif [ -n "$PKG_CONFIG_PATH" ]; then
    # If PKG_CONFIG_PATH is set, try pkg-config first with the custom path
    echo "PKG_CONFIG_PATH is set, trying pkg-config with custom path..."
fi

# First, try pkg-config with standard package names (unless already set via env vars)
if [ -z "$OPENCV_CFLAGS" ]; then
    if pkg-config --exists opencv4 2>/dev/null; then
        OPENCV_PKG="opencv4"
        echo "Found OpenCV via pkg-config (opencv4)"
    elif pkg-config --exists opencv 2>/dev/null; then
        OPENCV_PKG="opencv"
        echo "Found OpenCV via pkg-config (opencv)"
    else
        # pkg-config failed, try to find OpenCV manually (common on Jetson devices)
        echo "pkg-config could not find OpenCV, trying manual detection..."
        
        # Common OpenCV installation paths
        OPENCV_PATHS=(
            "/usr/local"
            "/usr"
            "/opt/opencv"
            "/usr/local/opencv"
        )
        
        OPENCV_INCLUDE=""
        OPENCV_LIB_DIR=""
        
        # Try to find OpenCV headers
        for base_path in "${OPENCV_PATHS[@]}"; do
            if [ -d "$base_path/include/opencv4" ]; then
                OPENCV_INCLUDE="-I$base_path/include/opencv4"
                echo "Found OpenCV headers at: $base_path/include/opencv4"
                break
            elif [ -d "$base_path/include/opencv2" ]; then
                OPENCV_INCLUDE="-I$base_path/include"
                echo "Found OpenCV headers at: $base_path/include"
                break
            fi
        done
        
        # Try to find OpenCV libraries
        for base_path in "${OPENCV_PATHS[@]}"; do
            if [ -d "$base_path/lib" ] && ls "$base_path/lib"/libopencv*.so* 1>/dev/null 2>&1; then
                OPENCV_LIB_DIR="-L$base_path/lib"
                echo "Found OpenCV libraries at: $base_path/lib"
                break
            fi
        done
        
        # If we found both headers and libs, construct flags manually
        if [ -n "$OPENCV_INCLUDE" ] && [ -n "$OPENCV_LIB_DIR" ]; then
            OPENCV_CFLAGS="$OPENCV_INCLUDE"
            # Link against common OpenCV libraries
            OPENCV_LIBS="$OPENCV_LIB_DIR -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
            echo "Using manually detected OpenCV paths"
        else
            # Last resort: try setting PKG_CONFIG_PATH for common locations
            for base_path in "${OPENCV_PATHS[@]}"; do
                if [ -d "$base_path/lib/pkgconfig" ]; then
                    export PKG_CONFIG_PATH="$base_path/lib/pkgconfig:$PKG_CONFIG_PATH"
                    echo "Added to PKG_CONFIG_PATH: $base_path/lib/pkgconfig"
                    
                    # Try pkg-config again with updated path
                    if pkg-config --exists opencv4 2>/dev/null; then
                        OPENCV_PKG="opencv4"
                        echo "Found OpenCV via pkg-config after updating PKG_CONFIG_PATH (opencv4)"
                        break
                    elif pkg-config --exists opencv 2>/dev/null; then
                        OPENCV_PKG="opencv"
                        echo "Found OpenCV via pkg-config after updating PKG_CONFIG_PATH (opencv)"
                        break
                    fi
                fi
            done
            
            # If still not found, give up
            if [ -z "$OPENCV_PKG" ] && [ -z "$OPENCV_CFLAGS" ]; then
                echo "Error: OpenCV is not installed or cannot be found."
                echo ""
                echo "Troubleshooting steps:"
                echo "1. Install OpenCV: sudo apt-get install libopencv-dev"
                echo "2. If OpenCV is installed but pkg-config can't find it:"
                echo "   - Find where OpenCV is installed: find /usr -name 'opencv2' 2>/dev/null"
                echo "   - Set PKG_CONFIG_PATH: export PKG_CONFIG_PATH=/path/to/opencv/lib/pkgconfig:\$PKG_CONFIG_PATH"
                echo "   - Or set environment variables:"
                echo "     export OPENCV_INCLUDE_DIR=/path/to/opencv/include"
                echo "     export OPENCV_LIB_DIR=/path/to/opencv/lib"
                echo "3. For Jetson devices, OpenCV is often in /usr/local"
                echo "   Try: export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:\$PKG_CONFIG_PATH"
                exit 1
            fi
        fi
    fi
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Build command
echo "Compiling ransac_bindings.cpp..."

# Construct compile command based on how OpenCV was detected
if [ -n "$OPENCV_PKG" ]; then
    # Use pkg-config
    OPENCV_CFLAGS=$(pkg-config --cflags $OPENCV_PKG)
    OPENCV_LIBS=$(pkg-config --libs $OPENCV_PKG)
fi

g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    $EIGEN_INCLUDE \
    $OPENCV_CFLAGS \
    ransac_bindings.cpp \
    -o ransac_bindings$(python3-config --extension-suffix) \
    $OPENCV_LIBS

if [ $? -eq 0 ]; then
    echo "Success! ransac_bindings$(python3-config --extension-suffix) has been created."
    echo "You can now import it in Python with: import ransac_bindings"
else
    echo "Build failed!"
    exit 1
fi

