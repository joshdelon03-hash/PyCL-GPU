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
    Demonstrates image sharpening using a Lanczos-based kernel.
    """
    print("--- GPGPU Framework Demonstration: Lanczos Image Sharpening ---")

    # The OpenCL kernel for Lanczos sharpening
    kernel_code = """
    #define M_PI 3.14159265358979323846f

    // Sinc function
    float sinc(float x) {
        if (x == 0.0f) {
            return 1.0f;
        }
        return sin(M_PI * x) / (M_PI * x);
    }

    // 1D Lanczos kernel
    float lanczos_kernel(float x, float a) {
        if (x > -a && x < a) {
            return sinc(x) * sinc(x / a);
        }
        return 0.0f;
    }

    __kernel void lanczos_sharpen(
        __read_only image2d_t src_img,
        __write_only image2d_t dst_img,
        const int width,
        const int height,
        const float amount,
        const float a) 
    {
        const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
        
        int2 pos = {get_global_id(0), get_global_id(1)};

        if (pos.x >= width || pos.y >= height) {
            return;
        }

        float4 original_color = read_imagef(src_img, sampler, pos);
        float4 blurred_color = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float total_weight = 0.0f;

        int radius = (int)ceil(a);

        for (int j = -radius; j <= radius; j++) {
            for (int i = -radius; i <= radius; i++) {
                float l_x = lanczos_kernel((float)i, a);
                float l_y = lanczos_kernel((float)j, a);
                float weight = l_x * l_y;

                if (weight > 0.0f) {
                    int2 neighbor_pos = pos + (int2)(i, j);
                    blurred_color += read_imagef(src_img, sampler, neighbor_pos) * weight;
                    total_weight += weight;
                }
            }
        }

        if (total_weight > 0.0f) {
            blurred_color /= total_weight;
        } else {
            blurred_color = original_color;
        }

        // Sharpening formula: sharpened = original + (original - blurred) * amount
        float4 sharpened_color = original_color + (original_color - blurred_color) * amount;
        
        write_imagef(dst_img, pos, sharpened_color);
    }
    """

    try:
        # 1. Load the image
        image_path = os.path.join("..", "example_images", "butterfly.png")
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        # Ensure image is 4-channel (BGRA)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        height, width, _ = img.shape
        print(f"Loaded image: {width}x{height}")

        # 2. Create the ParallelTask
        sharpen_task = ParallelTask(kernel_code, use_image_objects=True)

        # 3. Create device buffers
        src_buffer = DeviceBuffer.from_image(sharpen_task.ctx, img)
        dst_buffer = DeviceBuffer.empty_image(sharpen_task.ctx, (width, height))

        # 4. Execute the kernel
        sharpening_amount = 1.5 
        lanczos_a = 2.0 # a=2 is Lanczos-2
        global_size = (width, height)
        kernel_args = [
            src_buffer,
            dst_buffer,
            np.int32(width),
            np.int32(height),
            np.float32(sharpening_amount),
            np.float32(lanczos_a)
        ]

        print("Executing Lanczos sharpening kernel...")
        sharpen_task.execute(global_size, kernel_args)

        # 5. Read the result back
        result_img = dst_buffer.read_image()

        # 6. Save the output
        output_filename = os.path.join("..", "example_images", "butterfly_sharpened.png")
        cv2.imwrite(output_filename, result_img)
        print(f"Sharpened image saved to {output_filename}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
