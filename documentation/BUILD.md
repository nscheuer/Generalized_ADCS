# Build Instructions for Trajectory Planner

This document covers building the C++ trajectory planner module (`tplaunch`).

## Prerequisites

- CMake 3.16+
- Python 3.10+
- Armadillo library
- GCC 11+ (or Clang 14+)

## Basic Build

```bash
cd trajectory_planner
mkdir -p build && cd build
cmake ..
make -j$(nproc) tplaunch
```

## Build Acceleration Options

### 1. Ninja Build System (Recommended)

Ninja is faster than Make for incremental builds.

```bash
# Install Ninja
sudo apt install ninja-build

# Build with Ninja
cd trajectory_planner/build
rm -rf *
cmake -G Ninja ..
ninja tplaunch
```

### 2. ccache (Already Enabled)

ccache is automatically detected and used if installed. It dramatically speeds up rebuilds.

```bash
# Install ccache
sudo apt install ccache

# Verify it's being used (look for "Found ccache" in cmake output)
cmake ..
```

### 3. Precompiled Headers (PCH)

The build system uses precompiled headers to speed up compilation. Heavy includes like Armadillo and pybind11 are precompiled once and reused.

PCH is automatically enabled - no configuration needed.

### 4. mold Linker (Optional)

The mold linker is significantly faster than the default linker.

```bash
# Install mold
sudo apt install mold

# Use mold for linking (add to CMake command)
cmake -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=mold" ..
```

## Build Types

### Release Build (Default)
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Includes optimizations:
- `-O3` - Maximum optimization
- `-march=native` - CPU-specific optimizations
- `-flto` - Link-time optimization
- `-funroll-loops` - Loop unrolling

**Note:** `-ffast-math` is intentionally NOT used as it breaks numerical stability in the ALTRO trajectory optimization algorithm.

### Debug Build
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

## Verbosity Levels

The planner supports 5 verbosity levels:
- **0** - Silent (no output)
- **1** - Milestones only (start/end of major operations)
- **2** - Progress updates (iteration counts, convergence)
- **3** - Detailed debug output
- **4** - Expensive debug operations (may slow down execution)

Set verbosity in Python:
```python
planner.setVerbosity(2)  # Progress updates
```

## Troubleshooting

### ODR Warnings
The build may show One Definition Rule (ODR) warnings with LTO enabled. These are false positives from Armadillo template instantiation and are suppressed with `-Wno-odr`.

### Armadillo Fast-Math Warnings
These are suppressed via `ARMA_DONT_PRINT_FAST_MATH_WARNING` compile definition.

### Slow Initial Build
The first build is slow because:
1. FetchContent downloads pybind11 and Catch2
2. Large C++ files need compilation

Subsequent builds are much faster due to ccache and PCH.
