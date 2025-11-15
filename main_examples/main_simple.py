import numpy as np
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

def main():
    """
    Demonstrates the simplified workflow using the ParallelTask helper class.
    This script performs the same task as main.py but with much less boilerplate code.
    This version has been optimized to use float4 vectorization and persistent device buffers.
    """
    print("--- GPGPU Framework Demonstration: Refactored Workflow with Persistent Buffers (Vectorized) ---")

    # Define the OpenCL C kernel code as a string.
    kernel_code = """
    __kernel void vector_add(__global const float4 *a,
                             __global const float4 *b,
                             __global float4 *c)
    {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
    """

    try:
        # 1. Create the ParallelTask object. This compiles the kernel once.
        add_task = ParallelTask(kernel_code)

        # 2. Prepare host data and create persistent device buffers.
        #    These buffers live on the GPU and can be reused without re-allocation.
        n_elements = 1048576 # 2^20
        print(f"\nPreparing host data and creating persistent GPU buffers for {n_elements} elements...")
        
        a_host = np.arange(n_elements, dtype=np.float32)
        b_host = np.arange(n_elements, dtype=np.float32) * 2
        c_host_template = np.empty(n_elements, dtype=np.float32)

        # Create buffers on the device
        a_buffer = DeviceBuffer.from_numpy(add_task.ctx, a_host)
        b_buffer = DeviceBuffer.from_numpy(add_task.ctx, b_host)
        c_buffer = DeviceBuffer.empty_like(add_task.ctx, c_host_template)

        # 3. Execute the task using the pre-created buffers.
        global_size = (n_elements // 4,)
        kernel_args = [a_buffer, b_buffer, c_buffer]
        
        print("Executing kernel with persistent buffers...")
        add_task.execute(global_size, kernel_args)

        # 4. Read the result back from the output buffer.
        print("Reading result from GPU...")
        c_host = c_buffer.read()

        # In a real application, you could now update the input buffers and re-run
        # the kernel many times without the overhead of creating new buffers.
        # For example:
        # for i in range(10):
        #     a_host += 1
        #     a_buffer.write(a_host) # Update data on GPU
        #     add_task.execute(global_size, kernel_args) # Re-run kernel
        #     result = c_buffer.read() # Get new result
        #     print(f"Iteration {i} result: {result.flatten()[:4]}...")

        # 5. Verify the results.
        print("\n--- Verification ---")
        expected_result = a_host + b_host
        if np.allclose(c_host, expected_result):
            print("Success! The GPU result matches the CPU result.")
            print(f"First 10 results (flattened): {c_host.flatten()[:10]}")
        else:
            print("Error! The GPU result does not match the CPU result.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a working OpenCL driver installed for your hardware.")

if __name__ == "__main__":
    main()
