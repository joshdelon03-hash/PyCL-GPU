import numpy as np
import re
import amp
from .context import ComputeContext
from .buffer import DeviceBuffer

class ParallelTask:
    """
    A high-level wrapper to simplify executing a GPGPU task.
    This class is responsible for compiling a kernel and executing it with
    pre-allocated device buffers.
    """
    def __init__(self, kernel_code, use_image_objects=False):
        """
        Initializes the task by creating a compute context and compiling the kernel.
        """
        self.ctx = ComputeContext(use_image_objects=use_image_objects)
        self.kernel_code = kernel_code
        
        match = re.search(r'__kernel\s+void\s+(\w+)\s*\(', self.kernel_code)
        if not match:
            raise ValueError("Could not find a kernel function name in the provided code.")
        self.kernel_name = match.group(1)

        print(f"\nCompiling kernel '{self.kernel_name}'...")
        self.program = self.ctx.compile(self.kernel_code)

    def execute(self, global_size, kernel_args, local_size=None):
        """
        Executes the pre-compiled kernel with the given device buffers and scalars.
        """
        kernel_func = getattr(self.program, self.kernel_name)
        print(f"Executing kernel '{self.kernel_name}' with global_size {global_size}...")
        kernel_func(*kernel_args, global_size=global_size, local_size=local_size)

    def release(self):
        """
        Releases the underlying compute context.
        """
        if self.ctx:
            self.ctx.release()
            self.ctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

class MultiParallelTask:
    """
    The 'AMP' Upgraded Task Manager.
    Uses the Adaptive Multiprocessing Pool to orchestrate multiple GPU tasks
    across different CPU cores, while each core interacts with the GPU.
    
    This is useful for 'Ensemble' processing, where you run many slightly 
    different simulations in parallel.
    """
    def __init__(self, kernel_code, use_image_objects=False, pool_size=None):
        self.kernel_code = kernel_code
        self.use_image_objects = use_image_objects
        # Initialize the global multiprocessing pool
        self.pool = amp.GlobalMPPool(min_workers=pool_size)

    def _worker_execute(self, global_size, kernel_args, local_size=None):
        """
        Internal worker function that runs in a separate process.
        It must create its OWN OpenCL context because contexts are not 
        typically thread/process safe for sharing in this way.
        """
        # 1. Create a fresh context in this process
        task = ParallelTask(self.kernel_code, use_image_objects=self.use_image_objects)
        
        # 2. Convert any numpy arrays in kernel_args to DeviceBuffers for THIS context
        processed_args = []
        for arg in kernel_args:
            if isinstance(arg, np.ndarray):
                # We assume if the array is zeros/empty, it's an output buffer
                if np.all(arg == 0) and arg.size > 0:
                    processed_args.append(DeviceBuffer.empty_like(task.ctx, arg))
                else:
                    processed_args.append(DeviceBuffer.from_numpy(task.ctx, arg))
            else:
                # Keep scalars as-is
                processed_args.append(arg)

        # 3. Execute
        task.execute(global_size, processed_args, local_size=local_size)
        
        # 4. Cleanup (Security/Robustness)
        for arg in processed_args:
            if isinstance(arg, DeviceBuffer):
                arg.release()
        task.ctx.release()

        # We could read results back and return them here if needed.
        return True

    def run_async(self, global_size, kernel_args, local_size=None, success=None):
        """
        Submits a task to the multiprocessing pool.
        """
        amp.async_call(
            self._worker_execute,
            global_size, kernel_args, local_size,
            success=success
        )

    def wait_all(self):
        """Wait for all submitted tasks to complete."""
        self.pool.join()

    def shutdown(self):
        """Shutdown the underlying AMP pool."""
        self.pool.shutdown()
