import pyopencl as cl
import numpy as np

class DeviceBuffer:
    """
    A wrapper for an OpenCL memory object (buffer or image) on a compute device.
    """
    def __init__(self, context, cl_mem, shape, dtype, is_image=False):
        """
        Initializes the DeviceBuffer.
        """
        self.context = context
        self.cl_mem = cl_mem
        self.shape = shape
        self.dtype = dtype
        self.is_image = is_image
        self.size = cl_mem.size

    @classmethod
    def from_numpy(cls, context, arr):
        """
        Creates a DeviceBuffer from a NumPy array as a cl.Buffer.
        """
        flags = cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR
        cl_buffer = cl.Buffer(context.context, flags, hostbuf=arr)
        return cls(context, cl_buffer, arr.shape, arr.dtype)

    @classmethod
    def empty_like(cls, context, arr):
        """
        Creates an empty, READ_WRITE cl.Buffer with the same shape and dtype as a given NumPy array.
        """
        flags = cl.mem_flags.READ_WRITE
        cl_buffer = cl.Buffer(context.context, flags, size=arr.nbytes)
        return cls(context, cl_buffer, arr.shape, arr.dtype)

    @classmethod
    def from_image(cls, context, arr):
        """
        Creates a DeviceBuffer from a NumPy array as a cl.Image.
        """
        # For BGRA or RGBA, channel_order is BGRA/RGBA. For grayscale, it's R.
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                channel_order = cl.channel_order.BGRA # OpenCV uses BGRA
            else:
                raise ValueError("3D array must be 4-channel (BGRA/RGBA)")
        elif arr.ndim == 2:
            channel_order = cl.channel_order.R
        else:
            raise ValueError("Unsupported image dimensions")

        image_format = cl.ImageFormat(channel_order, cl.channel_type.UNORM_INT8)
        flags = cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR
        cl_image = cl.Image(context.context, flags, image_format, shape=(arr.shape[1], arr.shape[0]), hostbuf=arr)
        return cls(context, cl_image, arr.shape, arr.dtype, is_image=True)

    @classmethod
    def empty_image(cls, context, shape):
        """
        Creates an empty, WRITE_ONLY cl.Image.
        """
        image_format = cl.ImageFormat(cl.channel_order.BGRA, cl.channel_type.UNORM_INT8)
        flags = cl.mem_flags.WRITE_ONLY
        cl_image = cl.Image(context.context, flags, image_format, shape=(shape[0], shape[1]))
        return cls(context, cl_image, (shape[1], shape[0], 4), np.uint8, is_image=True)

    def read(self):
        """
        Reads data from a cl.Buffer to a NumPy array.
        """
        if self.is_image:
            raise TypeError("Cannot use read() on an image buffer. Use read_image().")
        host_array = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.context.queue, host_array, self.cl_mem)
        return host_array

    def read_image(self):
        """
        Reads data from a cl.Image to a NumPy array.
        """
        if not self.is_image:
            raise TypeError("Cannot use read_image() on a non-image buffer.")
        buffer = np.empty(self.shape, dtype=self.dtype)
        cl.enqueue_copy(self.context.queue, buffer, self.cl_mem, origin=(0, 0), region=(self.shape[1], self.shape[0]))
        return buffer

    def write(self, arr):
        """
        Writes data from a NumPy array to an existing cl.Buffer.
        """
        if self.is_image:
            raise TypeError("Cannot use write() on an image buffer.")
        if arr.shape != self.shape or arr.dtype != self.dtype:
            raise ValueError("Input array shape or dtype does not match buffer's.")
        cl.enqueue_copy(self.context.queue, self.cl_mem, arr)

