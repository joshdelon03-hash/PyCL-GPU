import numpy as np
from framework.context import ComputeContext
from framework.buffer import DeviceBuffer

def main():
    """
    Demonstrates the use of our custom GPGPU framework to perform 
    a parallel vector addition.
    """
    print("--- GPGPU Framework Demonstration: Vector Addition ---")
    
    try:
        # 1. Initialize our architecture's context.
        # This finds the GPU and sets up the command queue.
        ctx = ComputeContext()

        # 2. Define the OpenCL C kernel code as a string.
        # This is the program that will run on the GPU.
        kernel_code = """
        __kernel void vector_add(__global const float *a,
                                 __global const float *b,
                                 __global float *c)
        {
            // Get the unique ID of this specific thread
            int i = get_global_id(0);

            // Perform the addition for the element at this thread's ID
            c[i] = a[i] + b[i];
        }
        """

        # 3. Compile the kernel using our framework.
        print("\nCompiling OpenCL kernel...")
        program = ctx.compile(kernel_code)

        # 4. Prepare data on the CPU (Host).
        n_elements = 100000
        print(f"Preparing host data for {n_elements} elements...")
        a_host = np.arange(n_elements, dtype=np.float32)
        b_host = np.arange(n_elements, dtype=np.float32) * 2

        # 5. Create DeviceBuffers to manage data on the GPU (Device).
        print("Transferring data from CPU to GPU...")
        # input buffers are copied from the host
        a_device = DeviceBuffer.from_numpy(ctx, a_host)
        b_device = DeviceBuffer.from_numpy(ctx, b_host)
        # output buffer is created empty on the device
        c_device = DeviceBuffer.empty_like(ctx, a_host)

        # 6. Execute the kernel.
        print("Executing kernel on GPU...")
        # This looks for the 'vector_add' kernel in our compiled program
        # and calls it with the specified buffers and work sizes.
        program.vector_add(a_device, b_device, c_device, global_size=(n_elements,), local_size=None)

        # 7. Copy the result from the GPU back to the CPU.
        print("Transferring result from GPU to CPU...")
        c_host = c_device.to_numpy()

        # 8. Verify the results.
        print("\n--- Verification ---")
        expected_result = a_host + b_host
        if np.allclose(c_host, expected_result):
            print("Success! The GPU result matches the CPU result.")
            print(f"First 10 results: {c_host[:10]}")
        else:
            print("Error! The GPU result does not match the CPU result.")
            print(f"Actual:   {c_host[:10]}")
            print(f"Expected: {expected_result[:10]}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure you have a working OpenCL driver installed for your hardware (e.g., from NVIDIA, AMD, or Intel).")

if __name__ == "__main__":
    main()
