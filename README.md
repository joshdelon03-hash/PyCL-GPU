# PyCL-GPU: A GPGPU Framework for Python

PyCL-GPU is a streamlined, Python-based framework for General-Purpose computing on Graphics Processing Units (GPGPU). It abstracts away the boilerplate of PyOpenCL, providing a clean, modular API for numerical and image processing tasks.

## Key Features

- **Clean API**: Manage GPU contexts, buffers, and kernels with minimal code.
- **AMP Integrated**: Includes the **Adaptive Multiprocessing Pool (AMP)** to orchestrate parallel GPU tasks across all available CPU cores.
- **Speedmaxxing Methodology**: High-velocity optimizations including FP16 vectorization, Memory Grip alignment, and Radix Reordering.
- **Intelligent Routing**: A* PCIe Data Routing to avoid hardware bus congestion.
- **Cross-Platform**: Works with any OpenCL-compliant hardware (NVIDIA, AMD, Intel).

## Research Paper

The methodology and performance results of this framework are documented in the paper:
**"Speedmaxxing: Direct Hardware Access and Optimization for GPGPU using PyCL-GPU"** (Included as `paper.pdf`).

## Project Structure

- `framework/`: The core library (Context, Buffer, Program, Task management).
- `amp/`: Adaptive Multiprocessing Pool implementation.
- `kernels/`: OpenCL C kernel source files.
- `main_dpj.py`: Dot-Product Join + Radix Reordering demo.
- `main_astar.py`: A* PCIe Routing visualization demo.
- `main_neural.py`: Latency Predictor neural-driven scaling demo.
- `main.py`: General ensemble example using AMP.

## Installation

1. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. The `amp` library is now bundled directly in the `PyCL-GPU-main` folder. No separate installation is required.

## Quick Start

### Simple Task
```python
from framework.task import GeneralTask
# See main_simple.py for full context
```

### Multiprocessing Ensemble (AMP)
```python
from framework.task import MultiParallelTask
# Uses all CPU cores to feed the GPU
```

## License
Polyform Non-Commercial License 1.0.0.
"Personal use, teaching, and research are allowed. For-profit business use requires a separate license."
