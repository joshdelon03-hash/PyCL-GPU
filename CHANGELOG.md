# PyCL-GPU Changelog

All notable changes to this project will be documented in this file.

## [2026-03-10] - A* PCIe Data Traversal (Push Week Day 4)

### Added
- **A* Router (FP16)**: Implemented a high-speed routing kernel (`astar.cl`) that uses float16 (half) precision to calculate pathfinding heuristics across simulated PCIe lanes.
- **Hardware Dispatch**: The `AStarRouter` now automatically detects hardware capabilities, falling back to FP32 if `cl_khr_fp16` is not supported, ensuring "underhead" speed on compatible hardware (XLA/TPU/H100) while maintaining stability on legacy systems.
- **Congestion Mapping**: The `AStarRouter` treats the physical bus and memory buffers as a grid where "obstacles" are latency spikes or saturated buffers, providing a predictive routing layer for data.
- **Result Visualization**: Added `main_astar.py` to demonstrate and visualize the A* routing logic with real-time congestion heatmaps.

---

## [2026-03-09] - Neural-Driven Dynamic Scaling (Push Week Day 3)

### Added
- **Neural Monitor**: Implemented a `LatencyPredictor` in `framework/neural.py` using PyTorch to forecast task execution times based on data size, queue depth, and worker counts.
- **Dynamic Scaling**: The `AMP` pool now utilizes "Neural Scaling" to throttle or expand worker counts in real-time, preventing system unresponsiveness during high-pressure bursts.
- **Neural Stress Test**: Added `main_neural.py` to verify the robustness of the system under 2024x memory pressure tests.

---

## [2026-03-08] - Generalized AMP & Dot-Product Join (Push Week Day 2)

### Added
- **Universal Orchestrator**: Generalised the `AMP` pool to support any Python function, making it compatible with PyTorch, Scikit-learn, and custom algorithms beyond OpenCL.
- **Dot-Product Join (DPJ)**: Implemented the `RadixReorder` algorithm to optimize sparse vector operations, achieving significant cache-efficiency gains.
- **General Task API**: Introduced the `GeneralTask` wrapper for seamless parallelization of non-GPGPU payloads.

---

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
