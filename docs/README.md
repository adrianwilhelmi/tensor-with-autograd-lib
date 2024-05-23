# Tensor with Autograd Library
## Overview
PyTorch-like lightweight header-only tensor library with autograd system written in C++. It implements two main components:
1. **Tensor Object**: A versatile n-dimensional array structure supporting a variety of operations.
2. **Autograd System**: An automatic differentiation system to facilitate gradient-based machine learning algorithms.

## Functionalities
* **Tensor Operations**:
  - I/O operations (loading from and saving to images and videos).
  - Tensor creation functions (zeros, ones, random tensors).
  - Basic arithmetic operations and mathematical functions.
  - Conveniet slicing, shaping and indexing operations
  - All the necessary linear algebra operations (dot product, fast matrix multiplication).
  - Machine learning related operations (pooling operations, activation functions, convolution operations)

* **Autograd System**:
  - Automatic differentiation to compute gradients for tensor operations.
  - Function objects to encapsulate operations for gradient computation.

## Requirements
* Any linux distribution
* `clang`
* `make`
* OpenCV library (loading images and videos into tensors)

Executing `scripts/install_env.sh` script will detect your Linux distribution and install required software (clang, make, opencv). You can run `make install environment` if you have at least make installed on your system. To run tests and all analyzers you will also need `valgrind` and `GTest`, but they are not essential for the library so i didn't include them in the installation script.

## Installation, compilation and usage

To build and install the library to a desired directory with a preferred compiler use the command: ``` make install P=/desired/path/```. Programs using the allocator should be compiled with `clang -std=c++17 -Ipath/to/installed/headers -I/path/to/opencv4` flags. Additionally, don't forget to include `#include"tensor/tensor_lib.hpp"` in your C program using the library. For usage check out examples and documentation located in `docs/` directory.
