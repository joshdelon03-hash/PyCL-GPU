# PyCL-GPU: A GPGPU Framework for Python

PyCL-GPU is a streamlined, Python-based framework for General-Purpose computing on Graphics Processing Units (GPGPU). It abstracts away the boilerplate of PyOpenCL, providing a clean, modular API for numerical and image processing tasks.

## Key Features

- **Clean API**: Manage GPU contexts, buffers, and kernels with minimal code.
- **AMP Integrated**: Built-in support for the **Adaptive Multiprocessing Pool (AMP)** to orchestrate parallel GPU tasks across multiple CPU cores.
- **Cross-Platform**: Works with any OpenCL-compliant hardware (NVIDIA, AMD, Intel).

## Project Structure

- `framework/`: The core library (Context, Buffer, Program, Task management).
- `kernels/`: OpenCL C kernel source files (e.g., `core.cl`).
- `main_simple.py`: A basic vector addition example.
- `main.py`: An advanced "ensemble" example using multiprocessing (AMP).

## Installation

1. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Install the `amp` library (included in this repository's ecosystem):
   ```bash
   cd ../amp_library
   pip install -e .
   ```

## Quick Start

### Simple Task
```python
from framework.task import ParallelTask
task = ParallelTask(kernel_code)
task.execute(global_size=(N,), kernel_args=[a_gpu, b_gpu, c_gpu])
```

### Multiprocessing Ensemble
```python
from framework.task import MultiParallelTask
ensemble = MultiParallelTask(kernel_code)
ensemble.run_async(global_size=(N, N), kernel_args=[A, B, C, N])
ensemble.wait_all()
```

## License
MIT License.
