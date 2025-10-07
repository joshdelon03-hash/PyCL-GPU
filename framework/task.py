import numpy as np
import re
from .context import ComputeContext
from .buffer import DeviceBuffer

class OutputBuffer:
    """A simple placeholder class to specify the details of an output buffer."""
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype

class ParallelTask:
    """
    A high-level wrapper to simplify executing a GPGPU task.
    """
    def __init__(self, kernel_code):
        """
        Initializes the task by creating a compute context and compiling the kernel.
        """
        self.ctx = ComputeContext()
        self.kernel_code = kernel_code
        
        match = re.search(r'__kernel\s+void\s+(\w+)\s*\(', self.kernel_code)
        if not match:
            raise ValueError("Could not find a kernel function name in the provided code.")
        self.kernel_name = match.group(1)

        print(f"\nCompiling kernel '{self.kernel_name}'...")
        self.program = self.ctx.compile(self.kernel_code)

    def execute(self, global_size, kernel_args, local_size=None):
        """
        Executes the pre-compiled kernel with a flexible list of arguments.

        Args:
            global_size (tuple): The global work size for the kernel execution (e.g., (width, height)).
            kernel_args (list): A list of arguments for the kernel. Can contain:
                                - np.ndarray: For read-only input buffers.
                                - OutputBuffer: A placeholder for the write-only output buffer.
                                - int, float: Scalar arguments for the kernel.
            local_size (tuple, optional): The local work size. Defaults to None.

        Returns:
            np.ndarray: The result from the output buffer as a NumPy array.
        """
        print("Preparing device buffers and arguments...")
        
        processed_args = []
        output_buffer_obj = None

        for arg in kernel_args:
            if isinstance(arg, np.ndarray):
                processed_args.append(DeviceBuffer.from_numpy(self.ctx, arg))
            elif isinstance(arg, OutputBuffer):
                if output_buffer_obj is not None:
                    raise ValueError("Only one OutputBuffer is allowed per task execution.")
                template = np.empty(arg.shape, dtype=arg.dtype)
                output_buffer_obj = DeviceBuffer.empty_like(self.ctx, template)
                processed_args.append(output_buffer_obj)
            else: # It's a scalar value (int, float, etc.)
                processed_args.append(arg)

        if output_buffer_obj is None:
            raise ValueError("No OutputBuffer specified in kernel_args.")

        kernel_func = getattr(self.program, self.kernel_name)

        print(f"Executing kernel '{self.kernel_name}' with global_size {global_size}...")
        kernel_func(*processed_args, global_size=global_size, local_size=local_size)
        
        print("Transferring result from GPU to CPU...")
        return output_buffer_obj.to_numpy()