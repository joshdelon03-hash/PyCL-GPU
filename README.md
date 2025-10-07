# PyCL-GPU: A GPUMiner-Inspired GPGPU Framework

This project is a simple, Python-based framework for General-Purpose computing on Graphics Processing Units (GPGPU) that uses OpenCL as its backend. It is inspired by the modular architecture described in the "Parallel Data Mining on Graphics Processors" paper (GPUMiner), providing a structured way to offload parallel computations to hardware like GPUs and multi-core CPUs from vendors like AMD, Intel, and NVIDIA.

It is designed as a lightweight, understandable alternative for developers who want to explore GPGPU programming without relying on the proprietary CUDA ecosystem.

## Core Architecture

The framework is composed of several key components:

1.  **`ComputeContext`**: The central controller that finds and manages the OpenCL device.
2.  **`DeviceBuffer`**: A wrapper for GPU memory that handles data transfers.
3.  **`Program` & `Kernel`**: Classes that compile and execute kernel code on the device.
4.  **`ParallelTask`**: A high-level helper class that wraps all the above components into a simple interface, making it easy to accelerate a Python function. It handles kernel compilation upon creation.

    **Note on Error Handling:** The `ParallelTask` constructor compiles your kernel code immediately. If there is a syntax error in your OpenCL C code, it will raise a `pyopencl.LogicError` containing detailed build logs from the OpenCL compiler. This helps you quickly find and debug issues in your kernel.

## Prerequisites

*   Python 3.x
*   NumPy, PyOpenCL, and Pillow
*   A functioning OpenCL 1.2+ driver for your hardware.

## Installation

Navigate to the project directory and run:
```bash
pip install -r requirements.txt
```

## Project Structure

The repository is organized as follows:

```
PyCL-GPU/
├── framework/ # The core framework source code
│ ├── __init__.py
│ ├── context.py
│ ├── task.py
│ └── ...
├── main_simple.py # A basic "vector add" example
├── main_render.py # The advanced ray tracing example
├── kernels/ # (Optional) A good place for your .cl files
│ └── render.cl
├── requirements.txt
└── README.md
```

## Quick Start: How to Accelerate a Python Function

The `ParallelTask` class is the simplest way to use this framework. Here is the general pattern to follow:

### Step 1: Identify the Parallel Logic in Your Code

Look for loops where the **same operation** is performed on every element of a large array.

```python
# This is the kind of code you can accelerate:
for i in range(len(a)):
    c[i] = a[i] + b[i]
```

### Step 2: Write the Logic as an OpenCL C Kernel

Convert the body of the loop into an OpenCL C kernel function and save it as a Python string.

```python
kernel_code = """
__kernel void vector_add(__global const float *a, __global const float *b, __global float *c)
{
    int i = get_global_id(0); // Get the unique ID for this thread
    c[i] = a[i] + b[i];
}
"""
```

### Step 3: Use `ParallelTask` to Execute the Kernel

Use the `main_simple.py` file as a template. Create a `ParallelTask` with your kernel, then call `.execute()`.

```python
from framework.task import ParallelTask, OutputBuffer
import numpy as np

# Create the task (this compiles the kernel)
add_task = ParallelTask(kernel_code)

# Prepare your data as NumPy arrays
a_host = np.arange(100000, dtype=np.float32)
b_host = np.arange(100000, dtype=np.float32) * 2

# Execute the task
c_host = add_task.execute(
    global_size=(100000,),
    kernel_args=[
        a_host, 
        b_host, 
        OutputBuffer(a_host.shape) # Specify the output
    ]
)
```

### How Data is Handled: Input vs. Output

The `execute` method intelligently handles data transfer between your computer's RAM (the "host") and the GPU's memory (the "device") based on the type of arguments you provide.

*   **Input Buffers (Read-Only):** When you pass a standard **NumPy array** into the `kernel_args` list, the framework treats it as a read-only input. It is automatically copied to the device's memory before the kernel runs, but it is not copied back.

*   **Output Buffers (Write-Only):** To get data *back* from the kernel, you must use the `OutputBuffer` helper class.
    *   `OutputBuffer(shape, dtype=np.float32)`: This tells the framework to create an empty, write-only buffer on the device with the specified shape and data type.
    *   After your kernel finishes writing to this buffer, its contents are automatically copied back to the host and returned from the `.execute()` method as a new NumPy array.

> **Pro Tip: Loading Kernels from Files**
>
> For larger projects, embedding kernel code in Python strings can become unwieldy. We recommend saving your OpenCL code in separate `.cl` files. This enables syntax highlighting and better organization.
>
> You can easily load the kernel into your `ParallelTask` like this:
>
> ```python
> # Read the kernel code from an external file
> with open('kernels/my_kernel.cl', 'r') as f:
>     kernel_code = f.read()
>
> # Create the task as usual
> my_task = ParallelTask(kernel_code)
> ```

---

## Advanced Usage: A Simple Ray Tracer

To demonstrate the flexibility of the framework, the `main_render.py` script uses `ParallelTask` to render a 3D scene. This is a perfect example of a more complex GPGPU task.

### The Goal

The script creates a PNG image of a 3D sphere with simple lighting. It does this by calculating the color for every single pixel in parallel on the GPU.

### The Kernel (`render_kernel`)

The OpenCL C kernel for this task is more complex. Here are the key parts:

1.  **Arguments:** It takes an output pixel buffer, the image `width` and `height` (as scalars), and a buffer containing scene objects (a sphere).
2.  **Pixel-to-Ray Calculation:** Each thread gets its unique 2D pixel coordinate (`x`, `y`) and calculates a "ray" from a virtual camera through that pixel.
3.  **Ray-Sphere Intersection:** The kernel contains math to test if this ray intersects with the sphere in the scene.
4.  **Shading:** If the ray hits the sphere, the kernel calculates a simple color based on how the surface faces a light source. If it misses, it writes a background color.
5.  **Output:** The final color for the pixel is written to the output buffer.

    **Note on Kernel Functions:** It is best practice to use OpenCL's built-in vector types (`float3`, `float4`, etc.) and functions (`dot`, `normalize`, `length`, etc.) whenever possible. Many OpenCL drivers provide highly optimized implementations of these, and defining your own can lead to compiler errors.

### The Python Host Code (`main_render.py`)

The Python script orchestrates the process:

1.  **Setup:** It defines the image size and the scene data (a NumPy array for the sphere).
2.  **Task Creation:** It creates a `ParallelTask` with the rendering kernel code.
3.  **Argument Assembly:** It creates a list of arguments for the kernel, showing the power of the flexible `execute` method:
    *   An `OutputBuffer` defines the output image buffer.
    *   The `width` and `height` are passed as simple `int` scalars.
    *   The sphere's data is passed as a NumPy array.
4.  **Execution:** It calls `task.execute()` with a 2D `global_size` (`(width, height)`), which maps one thread to each pixel.
5.  **Saving the Image:** The resulting NumPy array is reshaped and saved as `render_output.png` using the Pillow library.

### How to Run the Render

1.  Make sure you have installed the dependencies (`pip install -r requirements.txt`).
2.  Run the script from your terminal:
    ```bash
    python main_render.py
    ```
    After it finishes, you will find a `render_output.png` file in the project directory.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.