# PyCL-GPU: A GPUMiner-Inspired GPGPU Framework

This project is a simple, Python-based framework for General-Purpose computing on Graphics Processing Units (GPGPU) that uses OpenCL as its backend. It is inspired by the modular architecture described in the "Parallel Data Mining on Graphics Processors" paper (GPUMiner), providing a structured way to offload parallel computations to hardware like GPUs and multi-core CPUs.

After a significant refactoring, the framework now uses a more efficient, stream-like pattern with persistent device buffers, minimizing CPU-GPU overhead for repeated tasks.

## Core Architecture

The framework is composed of several key components:

1.  **`ComputeContext`**: The central controller that finds and manages the OpenCL device and command queue.
2.  **`DeviceBuffer`**: A wrapper for persistent GPU memory. It handles buffer allocation and provides `read()` and `write()` methods to transfer data between the host (CPU) and device (GPU).
3.  **`Program`**: A class that compiles and wraps OpenCL kernel code.
4.  **`ParallelTask`**: A high-level helper class that encapsulates the above components. It compiles a kernel and provides a lean `execute()` method to run it using pre-allocated `DeviceBuffer` objects.

## Prerequisites

*   Python 3.x
*   A functioning OpenCL 1.2+ driver for your hardware.

## Installation

Navigate to the project directory and install the required packages:
```bash
pip install -r requirements.txt
```

## Quick Start: The Refactored Workflow

The recommended way to use the framework is to create persistent `DeviceBuffer` objects and reuse them, which is highly efficient.

### Step 1: Write an OpenCL C Kernel

Convert the parallel part of your logic into an OpenCL C kernel. Using vector types like `float4` is highly recommended for performance.

```python
# The kernel is defined as a Python string
kernel_code = """
__kernel void vector_add(__global const float4 *a,
                         __global const float4 *b,
                         __global float4 *c)
{
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""
```

### Step 2: Use `ParallelTask` to Orchestrate

The host code creates the task, allocates persistent buffers on the GPU, and then executes the kernel.

```python
import numpy as np
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

# 1. Create the task (this compiles the kernel once)
add_task = ParallelTask(kernel_code)

# 2. Prepare host data and create persistent GPU buffers
a_host = np.arange(1048576, dtype=np.float32)
b_host = np.arange(1048576, dtype=np.float32) * 2
c_host_template = np.empty_like(a_host)

# These buffers now live on the GPU
a_buffer = DeviceBuffer.from_numpy(add_task.ctx, a_host)
b_buffer = DeviceBuffer.from_numpy(add_task.ctx, b_host)
c_buffer = DeviceBuffer.empty_like(add_task.ctx, c_host_template)

# 3. Define execution parameters and kernel arguments
global_size = (a_host.size // 4,) # We use float4, so we need 1/4 the work-items
kernel_args = [a_buffer, b_buffer, c_buffer]

# 4. Execute the kernel
add_task.execute(global_size, kernel_args)

# 5. Read the result back from the GPU
c_host = c_buffer.read()

# Now c_host contains the result of the GPU computation
```

This pattern is extremely efficient for iterative tasks. You can call `a_buffer.write(new_data)` to update GPU memory and then call `add_task.execute(...)` again with almost no overhead.

## Examples Included

This project includes several examples demonstrating the framework's capabilities.

### `main_simple.py`

A "hello world" example that performs element-wise addition of two large arrays. This is the best starting point for understanding the basic, high-performance workflow.

### `main_primes.py`

A high-performance prime number calculator. It uses a parallel segmented sieve algorithm, a classic GPGPU pattern:
1.  The CPU finds a small list of "base primes".
2.  The GPU receives this list.
3.  Each GPU thread takes a base prime and marks all of its multiples in a large number range as "not prime".

This demonstrates a powerful optimization strategy: using the CPU for small pre-calculations to simplify and accelerate the massively parallel work on the GPU.

### `main_2squared.py`

A simple but compelling demonstration of continuous, stateful computation. It initializes a single number on the GPU and then repeatedly calls a kernel to square it in an endless loop. This showcases the efficiency of the persistent buffer model, as the data lives on the GPU and is modified with very little CPU overhead.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.