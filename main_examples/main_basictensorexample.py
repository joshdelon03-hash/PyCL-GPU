import numpy as np
import matplotlib.pyplot as plt
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

def main():
    """
    Demonstrates a basic tensor operation (matrix multiplication) and plots the result.
    """
    print("--- GPGPU Framework Demonstration: Basic Tensor Example ---")

    # Define the OpenCL C kernel code for matrix multiplication.
    kernel_code = """
    __kernel void mat_mul(__global const float *a,
                          __global const float *b,
                          __global float *c,
                          const int M,
                          const int N,
                          const int K)
    {
        int i = get_global_id(0); // Row index
        int j = get_global_id(1); // Col index

        if (i < M && j < K) {
            float sum = 0.0f;
            for (int l = 0; l < N; l++) {
                sum += a[i * N + l] * b[l * K + j];
            }
            c[i * K + j] = sum;
        }
    }
    """

    try:
        # 1. Create the ParallelTask object.
        mat_mul_task = ParallelTask(kernel_code)

        # 2. Prepare host data.
        M, N, K = 256, 256, 256
        print(f"\\nPreparing host data for {M}x{N} * {N}x{K} matrix multiplication...")
        
        a_host = np.random.rand(M, N).astype(np.float32)
        b_host = np.random.rand(N, K).astype(np.float32)
        c_host_template = np.empty((M, K), dtype=np.float32)

        # 3. Create device buffers.
        a_buffer = DeviceBuffer.from_numpy(mat_mul_task.ctx, a_host)
        b_buffer = DeviceBuffer.from_numpy(mat_mul_task.ctx, b_host)
        c_buffer = DeviceBuffer.empty_like(mat_mul_task.ctx, c_host_template)

        # 4. Execute the task.
        global_size = (M, K)
        kernel_args = [
            a_buffer, 
            b_buffer, 
            c_buffer,
            np.int32(M),
            np.int32(N),
            np.int32(K)
        ]
        
        print("Executing kernel...")
        mat_mul_task.execute(global_size, kernel_args)

        # 5. Read the result back.
        print("Reading result from GPU...")
        c_host = c_buffer.read()

        # 6. Verify the results.
        print("\\n--- Verification ---")
        expected_result = a_host @ b_host
        if np.allclose(c_host, expected_result, atol=1e-5):
            print("Success! The GPU result matches the CPU result.")
        else:
            print("Error! The GPU result does not match the CPU result.")

        # 7. Plot the result.
        print("Plotting the result matrix...")
        plt.figure(figsize=(8, 6))
        plt.imshow(c_host, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Value')
        plt.title('Result of Matrix Multiplication')
        plt.xlabel('Column')
        plt.ylabel('Row')
        
        # 8. Save the plot.
        output_filename = os.path.join("..", "example_images", "basic_tensor_example.png")
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")

    except Exception as e:
        print(f"\\nAn error occurred: {e}")
        print("Please ensure you have a working OpenCL driver installed for your hardware.")

if __name__ == "__main__":
    main()
