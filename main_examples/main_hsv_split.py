import sys
import os
import cv2
import numpy as np
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

# Add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    """
    Demonstrates converting an image to HSV and creating a split-view image
    with half RGB and half HSV using an OpenCL kernel.
    """
    print("--- GPGPU Framework Demonstration: RGB/HSV Split-View ---")

    # The OpenCL kernel for creating the split-view
    kernel_code = """
    __kernel void hsv_split_view(
        __read_only image2d_t rgb_img,
        __read_only image2d_t hsv_img,
        __write_only image2d_t dst_img,
        const int width,
        const int height)
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        
        int2 pos = {get_global_id(0), get_global_id(1)};

        if (pos.x >= width || pos.y >= height) {
            return;
        }

        float4 color;
        if (pos.x < width / 2) {
            // Left half: RGB
            color = read_imagef(rgb_img, sampler, pos);
        } else {
            // Right half: HSV
            color = read_imagef(hsv_img, sampler, pos);
        }
        
        write_imagef(dst_img, pos, color);
    }
    """

    try:
        # 1. Load the image
        image_path = os.path.join("..", "example_images", "butterfly.png")
        rgb_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if rgb_img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        height, width, channels = rgb_img.shape
        print(f"Loaded image: {width}x{height}")

        # 2. Convert to HSV
        hsv_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)

        # 3. Ensure both images are 4-channel BGRA for the kernel
        if channels == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2BGRA)
        
        # The HSV image needs to be converted to BGRA to be stored in a 4-channel image object
        hsv_img_bgra = cv2.cvtColor(hsv_img_bgr, cv2.COLOR_HSV2BGR_FULL)
        hsv_img_bgra = cv2.cvtColor(hsv_img_bgra, cv2.COLOR_BGR2BGRA)


        # 4. Create the ParallelTask
        split_task = ParallelTask(kernel_code, use_image_objects=True)

        # 5. Create device buffers
        rgb_buffer = DeviceBuffer.from_image(split_task.ctx, rgb_img)
        hsv_buffer = DeviceBuffer.from_image(split_task.ctx, hsv_img_bgra)
        dst_buffer = DeviceBuffer.empty_image(split_task.ctx, (width, height))

        # 6. Execute the kernel
        global_size = (width, height)
        kernel_args = [
            rgb_buffer,
            hsv_buffer,
            dst_buffer,
            np.int32(width),
            np.int32(height)
        ]

        print("Executing HSV split-view kernel...")
        split_task.execute(global_size, kernel_args)

        # 7. Read the result back
        result_img = dst_buffer.read_image()

        # 8. Save the output
        output_filename = os.path.join("..", "example_images", "butterfly_hsv_split.png")
        cv2.imwrite(output_filename, result_img)
        print(f"Split-view image saved to {output_filename}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
