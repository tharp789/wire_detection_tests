# Building ransac_bindings C++ Extension

This document explains how to compile the `ransac_bindings.cpp` file into a Python extension module (`.so` file) on Ubuntu.

## Prerequisites

Install the required system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libeigen3-dev \
    libopencv-dev \
    pkg-config
```

Install Python dependencies:

```bash
pip install pybind11
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Build Methods

### Method 1: Using the Build Script (Recommended)

Simply run the provided build script:

```bash
cd wire_detection_python
bash build_ransac_bindings.sh
```

This script will:
- Check for all required dependencies
- Automatically detect Eigen3 and OpenCV locations
- Compile the extension module
- Create the `.so` file with the correct Python extension suffix

### Method 2: Manual Compilation

If you prefer to compile manually, use this command:

```bash
cd wire_detection_python
g++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    -I/usr/include/eigen3 \
    $(pkg-config --cflags opencv4) \
    ransac_bindings.cpp \
    -o ransac_bindings$(python3-config --extension-suffix) \
    $(pkg-config --libs opencv4)
```

**Note:** If your system uses `opencv` instead of `opencv4`, replace `opencv4` with `opencv` in the pkg-config commands.

### Method 3: Using setup.py (Alternative)

You can also create a `setup.py` file for a more Pythonic build process:

```python
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup
import pkgconfig

ext_modules = [
    Pybind11Extension(
        "ransac_bindings",
        ["ransac_bindings.cpp"],
        include_dirs=['/usr/include/eigen3'],
        libraries=pkgconfig.parse('opencv4')['libraries'],
        library_dirs=pkgconfig.parse('opencv4')['library_dirs'],
        extra_compile_args=['-O3'],
    ),
]

setup(
    name="ransac_bindings",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
```

Then build with:

```bash
python3 setup.py build_ext --inplace
```

## Verification

After building, verify the module can be imported:

```bash
cd wire_detection_python
python3 -c "import ransac_bindings; print('Success!')"
```

## Troubleshooting

### Error: "g++: command not found"
- Install build tools: `sudo apt-get install build-essential`

### Error: "python3-config: command not found"
- Install Python dev headers: `sudo apt-get install python3-dev`

### Error: "pybind11 not found"
- Install pybind11: `pip install pybind11`

### Error: "Eigen3 not found"
- Install Eigen3: `sudo apt-get install libeigen3-dev`
- If installed in a non-standard location, adjust the `-I` include path

### Error: "opencv4 not found" or "OpenCV is not installed or pkg-config cannot find it"
- Install OpenCV: `sudo apt-get install libopencv-dev`
- If pkg-config can't find it, you may need to set `PKG_CONFIG_PATH`
- Try using `opencv` instead of `opencv4` in the pkg-config commands

#### Jetson Device Specific Issues
On Jetson devices (NVIDIA Jetson Nano, Xavier, etc.), OpenCV is often installed via JetPack in non-standard locations. The build script now automatically tries to detect OpenCV in common locations, but if it still fails:

1. **Find where OpenCV is installed:**
   ```bash
   find /usr -name "opencv2" 2>/dev/null
   find /usr/local -name "opencv2" 2>/dev/null
   ```

2. **Set PKG_CONFIG_PATH:**
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   # Then run the build script again
   ```

3. **Or use environment variables (recommended for Jetson):**
   ```bash
   export OPENCV_INCLUDE_DIR=/usr/local/include/opencv4  # or /usr/local/include
   export OPENCV_LIB_DIR=/usr/local/lib
   # Then run the build script
   ```

The build script will automatically use these environment variables if they are set.

### Error: "undefined reference" or linking errors
- Make sure all OpenCV libraries are properly linked
- Check that OpenCV was installed correctly: `pkg-config --modversion opencv4`

### Wrong Python version
- The `.so` file is built for the Python version used to run the build command
- Make sure you're using the same Python version (and virtual environment) when importing

## Output

The build process will create a file named something like:
- `ransac_bindings.cpython-310-x86_64-linux-gnu.so` (for Python 3.10 on x86_64 Linux)

This file should be in the same directory as `ransac_bindings.cpp` and can be imported directly in Python.

