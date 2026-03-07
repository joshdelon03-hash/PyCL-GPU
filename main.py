import numpy as np
import time
from framework.task import MultiParallelTask
from framework.buffer import DeviceBuffer

# Kernel for a simple matrix-matrix multiplication
KERNEL_CODE = """
__kernel void matrix_mul(__global const float *a, 
                         __global const float *b, 
                         __global float *c, 
                         const int N) 
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += a[row * N + k] * b[k * N + col];
    }
    c[row * N + col] = sum;
}
"""

def main():
    """
    Demonstrates the MultiParallelTask (AMP Integrated).
    Orchestrates multiple GPU matrix multiplications using 
    concurrent CPU processes.
    """
    print("--- PyCL-GPU: MultiParallelTask (AMP) Ensemble Example ---")
    
    # Problem size: N x N matrices
    N = 256
    num_simulations = 8
    
    print(f"Running an ensemble of {num_simulations} matrix multiplications...")
    print(f"Matrix size: {N}x{N}")

    # 1. Initialize the MultiParallelTask (Uses AMP pool)
    ensemble = MultiParallelTask(KERNEL_CODE)

    # 2. A callback function to be called in the main process
    # when a task completes.
    def on_complete(result):
        # Result is True in this simple version
        pass

    start_time = time.time()

    # 3. Fire off the ensemble of tasks
    # Each task runs its own OpenCL context in a separate process
    for i in range(num_simulations):
        # Note: We generate random data in this main process 
        # and pass it down for simplicity in this demo.
        a = np.random.rand(N, N).astype(np.float32)
        b = np.random.rand(N, N).astype(np.float32)
        
        # We'll pass the numpy arrays directly. The _worker_execute 
        # inside MultiParallelTask will handle context creation.
        ensemble.run_async(
            global_size=(N, N),
            kernel_args=[a, b, np.zeros((N, N), dtype=np.float32), np.int32(N)],
            success=on_complete
        )

    print("Tasks submitted to the AMP pool. Waiting for GPU results...")
    ensemble.wait_all()

    duration = time.time() - start_time
    print(f"--- All tasks completed in {duration:.4f}s ---")

if __name__ == "__main__":
    # Note: On Windows, the 'if __name__ == "__main__":' guard 
    # is MANDATORY for multiprocessing.
    main()
