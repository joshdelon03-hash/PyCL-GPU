# PyCL-GPU Changelog

All notable changes to this project will be documented in this file.

## [2026-03-08] - The "Security" & Robustness Update (Push Week Day 1)

### Added
- **Resource Management**: Added explicit `release()` methods and Context Manager support (`with` statement) to `ComputeContext`, `DeviceBuffer`, and `ParallelTask`.
- **AMP Pool Shutdown**: Added `shutdown()` to `MultiParallelTask` and `MultiprocessingPool` for clean exits.
- **Worker Health Checks**: The `AMP` pool now monitors and automatically restarts worker processes if they crash, ensuring high-reliability during large runs.
- **Memory Protection**: Implemented `max_tasks_per_worker` in `MultiprocessingPool` to prevent long-term memory accumulation (leaks) by periodically refreshing worker processes.

### Changed
- Updated `MultiParallelTask` worker execution to proactively release GPU buffers and contexts after each task.
- Updated `main.py` and `main_simple.py` to demonstrate clean resource cleanup.

---

## [Pre-Push Week] - Foundation Era
- **Core Architecture**: Transitioned from raw PyOpenCL to a modular `framework/` structure.
- **AMP Integration**: Developed the Adaptive Multiprocessing Pool to orchestrate GPU tasks across multiple CPU cores.
- **Kernel Consolidation**: Standardized kernel management and simplified task execution.
- **The Dream**: Evolved from early experiments with Brook and GPUMiner into a streamlined GPGPU framework for Python.
