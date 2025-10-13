import numpy as np
import time
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

def main():
    """
    An application that starts with the number 2 and continuously squares it on the GPU.
    """
    print("--- 2 Squared: Continuous GPU Squaring ---")
    print("Press Ctrl+C to stop.")

    # The kernel takes a buffer containing a single double-precision float
    # and squares it in place.
    kernel_code = """
    __kernel void square_kernel(__global double *value)
    {
        // We only have one work-item, so we don't need to check the global_id.
        value[0] = value[0] * value[0];
    }
    """

    try:
        # 1. Create the ParallelTask. This compiles the kernel.
        square_task = ParallelTask(kernel_code)

        # 2. Prepare the initial host data and create the persistent device buffer.
        #    We use float64 (a double) because the numbers will get very large, very fast.
        host_data = np.array([2.0], dtype=np.float64)
        
        # This buffer will live on the GPU and be modified in a loop.
        data_buffer = DeviceBuffer.from_numpy(square_task.ctx, host_data)
        print(f"Starting value: {host_data[0]}\n")
        time.sleep(1)

        # 3. Start the endless computation loop.
        while True:
            # Execute the kernel. The global size is 1 as we only have one number.
            square_task.execute((1,), [data_buffer])

            # Read the result back from the GPU.
            result = data_buffer.read()

            current_value = result[0]
            print(current_value)

            # If the number becomes infinite or NaN (Not a Number), stop.
            if np.isinf(current_value) or np.isnan(current_value):
                print("\nValue has overflowed to infinity. Stopping.")
                break

            # Pause briefly to make the output readable.
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nComputation stopped by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
