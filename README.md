# PyCL-GPU: A GPGPU Framework for Python

This project is a simple, Python-based framework for General-Purpose computing on Graphics Processing Units (GPGPU) that uses OpenCL as its backend. It provides a structured way to offload parallel computations to hardware like GPUs and multi-core CPUs.

The framework has been refactored to support both buffer-based and image-based computations, making it suitable for a wider range of GPGPU tasks, from numerical processing to image manipulation.

## Core Architecture

The framework is composed of several key components:

1.  **`ComputeContext`**: The central controller that finds and manages the OpenCL device and command queue.
2.  **`DeviceBuffer`**: A wrapper for GPU memory, supporting both linear buffers and 2D image objects. It handles memory allocation and data transfer between the host (CPU) and device (GPU).
3.  **`Program`**: A class that compiles and wraps OpenCL kernel code.
4.  **`ParallelTask`**: A high-level helper class that encapsulates the above components. It compiles a kernel and provides a lean `execute()` method to run it.

## Prerequisites

*   Python 3.x
*   A functioning OpenCL 1.2+ driver for your hardware. For image processing examples, image support is required.

## Installation

Navigate to the project directory and install the required packages:
```bash
pip install -r requirements.txt
```

## How to Run the Examples

All examples are located in the `main_examples/` directory. To run an example, navigate to the project root and execute the script:

```bash
python main_examples/main_simple.py
python main_examples/main_lanczos.py
```

Output images from the examples will be saved in the `example_images/` directory.

## Examples Included

This project includes several examples demonstrating the framework's capabilities.

### Numerical Examples

*   **`main_simple.py`**: A "hello world" example that performs element-wise addition of two large arrays. This is the best starting point for understanding the basic workflow.
*   **`main_primes.py`**: A high-performance prime number calculator using a parallel segmented sieve algorithm.
*   **`main_2squared.py`**: A demonstration of continuous, stateful computation, where a number on the GPU is repeatedly squared.
*   **`main_basictensorexample.py`**: A basic tensor (matrix multiplication) example that plots the resulting matrix using `matplotlib`.

### Image Processing Examples

*   **`main_lanczos.py`**: Demonstrates image sharpening using a Lanczos-based convolution kernel. This example showcases the framework's support for OpenCL image objects.
*   **`main_hsv_split.py`**: Converts an image to the HSV color space and creates a split-view image with the original RGB on the left and HSV on the right.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
