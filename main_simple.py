import numpy as np
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

def main():
    """
    The 'Hello World' of PyCL-GPU.
    Performs element-wise addition of two large vectors.
    """
    print("--- PyCL-GPU: Simple Vector Addition Example ---")

    # 1. Define the OpenCL Kernel
    kernel_code = """
    __kernel void add_vectors(__global const float *a,
                             __global const float *b,
                             __global float *c)
    {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
    """

    # 2. Setup Task and Data
    N = 1000000
    print(f"Adding two vectors of size {N}...")
    
    a_host = np.random.rand(N).astype(np.float32)
    b_host = np.random.rand(N).astype(np.float32)

    # 3. Create the Task
    task = ParallelTask(kernel_code)

    # 4. Create Device Buffers
    # We use the task's context to allocate memory on the GPU
    a_gpu = DeviceBuffer.from_numpy(task.ctx, a_host)
    b_gpu = DeviceBuffer.from_numpy(task.ctx, b_host)
    c_gpu = DeviceBuffer.empty_like(task.ctx, a_host)

    # 5. Execute
    task.execute(global_size=(N,), kernel_args=[a_gpu, b_gpu, c_gpu])

    # 6. Read back and Verify
    result = c_gpu.read()
    
    if np.allclose(result, a_host + b_host):
        print("Success! GPU result matches CPU.")
    else:
        print("Error! Mismatch found.")

if __name__ == "__main__":
    main()
