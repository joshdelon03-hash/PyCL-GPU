import pyopencl as cl
import numpy as np

class DeviceBuffer:
    """
    A wrapper for an OpenCL memory buffer on a compute device.

    This class handles the allocation of memory on the device and the transfer
    of data between the host (CPU) and the device (GPU).
    """
    def __init__(self, context, cl_buffer, shape, dtype):
        """
        Initializes the DeviceBuffer. This is typically called via a classmethod factory.
        """
        self.context = context
        self.cl_buffer = cl_buffer
        self.shape = shape
        self.dtype = dtype
        self.size = cl_buffer.size

    @classmethod
    def from_numpy(cls, context, arr):
        """
        Factory method to create a DeviceBuffer from a NumPy array.
        The buffer is created as READ_WRITE and data is copied from the host.

        Args:
            context (ComputeContext): The compute context.
            arr (np.ndarray): The NumPy array to copy to the device.

        Returns:
            DeviceBuffer: A new buffer on the device containing the data.
        """
        flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        cl_buffer = cl.Buffer(context.context, flags, hostbuf=arr)
        return cls(context, cl_buffer, arr.shape, arr.dtype)

    @classmethod
    def empty_like(cls, context, arr):
        """
        Factory method to create an empty, READ_WRITE DeviceBuffer on the device 
        with the same shape and dtype as a given NumPy array.

        Args:
            context (ComputeContext): The compute context.
            arr (np.ndarray): The NumPy array to model the shape and type from.

        Returns:
            DeviceBuffer: A new, empty buffer on the device.
        """
        flags = cl.mem_flags.READ_WRITE
        cl_buffer = cl.Buffer(context.context, flags, size=arr.nbytes)
        return cls(context, cl_buffer, arr.shape, arr.dtype)

    def read(self):
        """
        Reads the data from the device (GPU) back to the host (CPU) and
        returns it as a NumPy array.

        Returns:
            np.ndarray: The data from the device as a NumPy array.
        """
        host_array = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.context.queue, host_array, self.cl_buffer)
        return host_array

    def write(self, arr):
        """
        Writes data from a NumPy array on the host (CPU) to this existing
        device buffer (GPU).

        Args:
            arr (np.ndarray): The source NumPy array.
        """
        if arr.shape != self.shape or arr.dtype != self.dtype:
            raise ValueError("Input array shape or dtype does not match buffer's.")
        # Blocking call to ensure data is written before any kernel uses it.
        cl.enqueue_copy(self.context.queue, self.cl_buffer, arr)
