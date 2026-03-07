// The main rendering kernel.
// Executed for every pixel to calculate its final color.
__kernel void render_kernel(__global uchar4 *output_pixels,
                            int width,
                            int height,
                            __global const float4 *sphere)
{
    // Get the 2D coordinates of the pixel this thread is responsible for
    int x = get_global_id(0);
    int y = get_global_id(1);
    int i = y * width + x;

    // Camera and screen setup (simple perspective)
    float aspect_ratio = (float)width / (float)height;
    float fov_adjust = tan(3.14159f * 0.5f * 0.5f); // 90 degree FOV
    
    float sensor_x = ((((float)x + 0.5f) / (float)width) * 2.0f - 1.0f) * aspect_ratio * fov_adjust;
    float sensor_y = (1.0f - (((float)y + 0.5f) / (float)height) * 2.0f) * fov_adjust;

    // Define the ray for this pixel
    float3 ray_origin = (float3)(0.0f, 0.0f, 0.0f);
    float3 ray_dir = normalize((float3)(sensor_x, sensor_y, -1.0f));

    // Sphere intersection test
    float3 sphere_center = (float3)(sphere[0].x, sphere[0].y, sphere[0].z);
    float sphere_radius = sphere[0].w;

    float3 oc = ray_origin - sphere_center;
    float b = 2.0f * dot(oc, ray_dir);
    float c = dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b*b - 4.0f*c;

    // Final pixel color (RGBA)
    uchar4 color = (uchar4)(20, 20, 40, 255); // Background color

    if (discriminant > 0.0f) {
        // The ray hit the sphere
        float t = (-b - sqrt(discriminant)) / 2.0f;
        if (t > 0.0f) { // Check if intersection is in front of the camera
            float3 hit_point = ray_origin + ray_dir * t;
            float3 normal = normalize(hit_point - sphere_center);
            // Simple lighting: color based on the surface normal
            float light = dot(normal, normalize((float3)(0.5f, 0.7f, 1.0f)));
            light = max(0.1f, light); // Ambient light
            color = (uchar4)(255 * light, 100 * light, 100 * light, 255);
        }
    }

    output_pixels[i] = color;
}