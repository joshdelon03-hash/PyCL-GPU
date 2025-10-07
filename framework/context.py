import pyopencl as cl
from .program import Program

class ComputeContext:
    """
    The CPU Controller for our GPGPU architecture.

    Manages the OpenCL context, device, and command queue, acting as the 
    central point for device interaction and resource management.
    """
    def __init__(self, device_type="GPU"):
        """
        Initializes the context by finding and selecting an OpenCL-capable device.

        Args:
            device_type (str): The preferred device type to use ("GPU" or "CPU"). 
                               Defaults to "GPU".
        """
        self.context = None
        self.queue = None
        self.device = None

        print("--- Initializing ComputeContext ---")
        try:
            target_device_type = cl.device_type.GPU if device_type == "GPU" else cl.device_type.CPU

            # Find the first platform that has a device of the desired type
            for platform in cl.get_platforms():
                try:
                    devices = platform.get_devices(device_type=target_device_type)
                    if devices:
                        self.device = devices[0] # Select the first available device
                        break
                except cl.RuntimeError:
                    # This can happen if a platform has no devices of the requested type
                    continue
            
            if not self.device:
                raise RuntimeError(f"No OpenCL devices of type '{device_type}' found.")

            # Create a context for the selected device
            self.context = cl.Context([self.device])
            
            # Create a command queue to send instructions to the device
            self.queue = cl.CommandQueue(self.context)

            print(f"Successfully initialized context on device: {self.device.name}")

        except cl.Error as e:
            print(f"OpenCL Error during initialization: {e}")
            raise
        except Exception as e:
            print(f"An error occurred during ComputeContext initialization: {e}")
            raise

    def compile(self, kernel_code):
        """
        Compiles OpenCL C code and returns a Program object.

        Args:
            kernel_code (str): A string containing the OpenCL C source code.

        Returns:
            Program: A Program object that can be used to execute kernels.
        """
        cl_program = cl.Program(self.context, kernel_code).build()
        return Program(self, cl_program)