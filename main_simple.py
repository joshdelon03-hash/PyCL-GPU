import numpy as np
from framework.task import ParallelTask

def main():
    """
    Demonstrates the simplified workflow using the ParallelTask helper class.
    This script performs the same task as main.py but with much less boilerplate code.
    """
    print("--- GPGPU Framework Demonstration: Simplified Workflow with ParallelTask ---")

    # Define the OpenCL C kernel code as a string.
    # The task is the same: element-wise addition of two arrays.
    kernel_code = """
    __kernel void vector_add(__global const float *a,
                             __global const float *b,
                             __global float *c)
    {
        int i = get_global_id(0);
        c[i] = a[i] + b[i];
    }
    """

    try:
        # 1. Create the ParallelTask object with the kernel code.
        #    In the background, this initializes the ComputeContext and compiles the kernel.
        add_task = ParallelTask(kernel_code)

        # 2. Prepare data on the CPU (Host).
        n_elements = 100000
        print(f"\nPreparing host data for {n_elements} elements...")
        a_host = np.arange(n_elements, dtype=np.float32)
        b_host = np.arange(n_elements, dtype=np.float32) * 2

        # 3. Execute the task.
        #    This single call handles buffer creation, data transfers to and from the GPU, 
        #    kernel execution, and returning the final result.
        c_host = add_task.execute([a_host, b_host], output_shape=(n_elements,))

        # 4. Verify the results.
        print("\n--- Verification ---")
        expected_result = a_host + b_host
        if np.allclose(c_host, expected_result):
            print("Success! The GPU result matches the CPU result.")
            print(f"First 10 results: {c_host[:10]}")
        else:
            print("Error! The GPU result does not match the CPU result.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a working OpenCL driver installed for your hardware.")

if __name__ == "__main__":
    main()
