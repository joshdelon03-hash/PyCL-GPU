// OpenCL Kernel for Day 4: A* PCIe Data Routing
// Optimization: Dynamic Precision (fp16 fallback to fp32)

#ifdef USE_FP16
    #pragma OPENCL EXTENSION cl_khr_fp16 : enable
    typedef half T;
#else
    typedef float T;
#endif

__kernel void pcie_astar_router(
    __global const T *lane_congestion, // The "Map" (Bus Saturation)
    __global const int2 *start_end,    // Start/Target Lanes
    __global float *heuristics,        // Output H values
    const int width,
    const int height)
{
    int gid = get_global_id(0);
    int2 pos = (int2)(gid % width, gid / width);
    int2 target = start_end[1];

    // Obstacle Check: If congestion is > 0.9 (90%), the lane is "saturated"
    T weight = lane_congestion[gid];
    if (weight > (T)0.9) {
        heuristics[gid] = -1.0f; // Mark as "Wall"
        return;
    }

    // Heuristic (h): Using dynamic precision to calculate Manhattan Distance
    T dx = (T)abs(pos.x - target.x);
    T dy = (T)abs(pos.y - target.y);
    
    // Day 4 Logic: Adding weight to the heuristic based on bus noise
    T h_score = (dx + dy) * ((T)1.0 + weight);

    // Write back to 32-bit for host-side logic stack (Day 3 stability)
    heuristics[gid] = (float)h_score;
}