import numpy as np
from framework.task import ParallelTask, OutputBuffer
from PIL import Image

def main():
    """
    Demonstrates the advanced usage of the ParallelTask class to render a simple 3D scene.
    """
    print("--- GPGPU Framework: Simple Ray Tracer ---")

    # 1. Define Image and Scene Parameters
    width, height = 800, 600
    
    # Define a sphere as [center_x, center_y, center_z, radius]
    # We add padding (w component) to make the data structure align to 16 bytes, which is often faster on GPUs.
    sphere = np.array([0.0, 0.0, -5.0, 1.0], dtype=np.float32)

    try:
        # 2. Load the OpenCL C kernel from the external .cl file
        print("Loading and compiling kernel from 'kernels/render.cl'...")
        with open('kernels/render.cl', 'r') as f:
            kernel_code = f.read()
        
        # 3. Create and execute the ParallelTask
        render_task = ParallelTask(kernel_code)
        
        # Define the arguments for the kernel
        kernel_args = [
            OutputBuffer((height, width, 4), dtype=np.uint8), # The output image buffer
            width,                                           # Image width
            height,                                          # Image height
            sphere                                           # Our scene object(s)
        ]

        # Execute the task. The global_size is the total number of pixels.
        print("Executing render task on the GPU...")
        image_data = render_task.execute(global_size=(width, height), kernel_args=kernel_args)

        # 4. Save the output image
        image = Image.fromarray(image_data, 'RGBA')
        image.save("render_output.png")
        print("\nSuccess! Image saved as render_output.png")

    except FileNotFoundError:
        print("\nError: Could not find the kernel file 'kernels/render.cl'.")
        print("Please ensure the file exists and you are running the script from the project root.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()