import numpy as np
import re
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

        Args:
            global_size (tuple): The global work size for the kernel execution.
            kernel_args (list): A list of arguments for the kernel. Must contain:
                                - DeviceBuffer objects for buffer arguments.
                                - int, float, etc. for scalar arguments.
            local_size (tuple, optional): The local work size. Defaults to None.
        """
        kernel_func = getattr(self.program, self.kernel_name)

        # The user's Program class (in program.py) likely handles the details of
        # unpacking the arguments. We pass the list of DeviceBuffers and scalars
        # directly to it, preserving the original calling convention.
        print(f"Executing kernel '{self.kernel_name}' with global_size {global_size}...")
        kernel_func(*kernel_args, global_size=global_size, local_size=local_size)