import pyopencl as cl
import numpy as np
from .buffer import DeviceBuffer

class Kernel:
    """
    A wrapper for an executable OpenCL kernel.
    This class is responsible for executing a specific kernel function from a compiled program.
    It is not typically instantiated directly, but rather through a Program object.
    """
    def __init__(self, context, cl_kernel):
        self.context = context
        self.cl_kernel = cl_kernel

    def __call__(self, *args, global_size, local_size=None):
        """
        Executes the OpenCL kernel on the device.

        Args:
            *args: Arguments to pass to the kernel. These can be DeviceBuffer objects
                   or other standard Python types (e.g., int, float).
            global_size (tuple): The global work size, defining the total number of work-items.
            local_size (tuple, optional): The local work size (work-items per work-group). 
                                        If None, OpenCL determines it. Defaults to None.
        """
        # Unpack DeviceBuffer objects and convert scalar types for the OpenCL API call
        processed_args = []
        for arg in args:
            if isinstance(arg, DeviceBuffer):
                processed_args.append(arg.cl_mem)
            elif isinstance(arg, int):
                processed_args.append(np.int32(arg))
            elif isinstance(arg, float):
                processed_args.append(np.float32(arg))
            else:
                processed_args.append(arg)
        
        # Execute the kernel with the provided queue, work sizes, and arguments
        self.cl_kernel(self.context.queue, global_size, local_size, *processed_args)


class Program:
    """
    A wrapper for a compiled OpenCL program.
    This class acts as a factory for Kernel objects contained within the program.
    """
    def __init__(self, context, cl_program):
        self.context = context
        self.cl_program = cl_program

    def __getattr__(self, name):
        """
        Dynamically gets a kernel function from the compiled program by its name.
        This allows for an intuitive API like: `program.my_kernel_name(...)`
        """
        try:
            cl_kernel = getattr(self.cl_program, name)
            return Kernel(self.context, cl_kernel)
        except cl.LogicError:
            raise AttributeError(f"Kernel '{name}' not found in the compiled program.")
