import numpy as np
import heapq
import os
from ..context import ComputeContext
from ..buffer import DeviceBuffer

class AStarRouter:
    """
    Day 4: A* PCIe Data Router.
    Uses an OpenCL kernel (fp16) to calculate pathfinding heuristics 
    across a simulated grid of PCIe lanes and buffers.
    """
    def __init__(self, width=64, height=64):
        self.width = width
        self.height = height
        self.grid_size = width * height
        
        # 1. Initialize Compute Context
        self.ctx = ComputeContext()
        
        # 2. Hardware Dispatch: Check for fp16 support
        extensions = self.ctx.device.extensions
        self.use_fp16 = "cl_khr_fp16" in extensions
        
        build_options = []
        if self.use_fp16:
            print("  [Hardware Dispatch] Using FP16 (cl_khr_fp16 enabled)")
            build_options = ["-D USE_FP16"]
            self.dtype = np.float16
        else:
            print("  [Hardware Dispatch] Falling back to FP32 (fp16 not supported)")
            self.dtype = np.float32

        # 3. Load and Compile the A* Kernel
        kernel_path = os.path.join(os.path.dirname(__file__), "..", "..", "kernels", "astar.cl")
        with open(kernel_path, 'r') as f:
            self.kernel_code = f.read()
        self.program = self.ctx.compile(self.kernel_code, options=build_options)
        
        # 4. Setup Buffers
        # Lane Congestion (Dynamic Precision: fp16 or fp32)
        self.congestion = np.zeros(self.grid_size, dtype=self.dtype)
        self.d_congestion = DeviceBuffer.empty_like(self.ctx, self.congestion)
        
        # Start/End Lanes (int2)
        self.start_end = np.zeros(2, dtype=[('x', np.int32), ('y', np.int32)])
        self.d_start_end = DeviceBuffer.empty_like(self.ctx, self.start_end)
        
        # Output Heuristics (Always float32 for stability in the main stack)
        self.heuristics = np.zeros(self.grid_size, dtype=np.float32)
        self.d_heuristics = DeviceBuffer.empty_like(self.ctx, self.heuristics)

    def update_congestion(self, congestion_map):
        """Updates the internal congestion map with real-time bus noise."""
        if congestion_map.size != self.grid_size:
            raise ValueError("Congestion map size mismatch.")
        self.congestion = congestion_map.astype(self.dtype)
        self.d_congestion.write(self.congestion)

    def calculate_heuristics_gpu(self, start_pos, end_pos):
        """
        Dispatches the A* kernel to calculate H-scores for the entire grid.
        This is the "Day 4" Optimization - offloading the guess to the GPU.
        """
        self.start_end[0] = start_pos
        self.start_end[1] = end_pos
        self.d_start_end.write(self.start_end)
        
        # Execute Kernel
        self.program.pcie_astar_router(
            self.d_congestion,
            self.d_start_end,
            self.d_heuristics,
            self.width,
            self.height,
            global_size=(self.grid_size,)
        )
        
        # Read back results
        self.heuristics = self.d_heuristics.read()
        return self.heuristics

    def find_path(self, start, end):
        """
        Performs the A* search on the CPU using GPU-accelerated heuristics.
        """
        # 1. Get GPU heuristics
        h_map = self.calculate_heuristics_gpu(start, end)
        
        # 2. A* Core Logic
        # Priority queue stores: (f_score, position)
        open_list = [(0.0, start)]
        came_from = {}
        g_score = {start: 0.0}
        
        while open_list:
            current_f, current = heapq.heappop(open_list)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Neighbors (Up, Down, Left, Right)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    idx = neighbor[1] * self.width + neighbor[0]
                    h = h_map[idx]
                    
                    if h < 0: # Obstacle / Saturated Lane
                        continue
                        
                    # Base cost + congestion penalty
                    # In Day 4 logic, the "congestion" is partially baked into h by the kernel,
                    # but we also add it to g to ensure the path doesn't just "guess" around it.
                    move_cost = 1.0 + float(self.congestion[idx])
                    tentative_g = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + h
                        heapq.heappush(open_list, (f_score, neighbor))
                        
        return None # No path found

    def release(self):
        """Cleanup OpenCL resources."""
        self.d_congestion.release()
        self.d_start_end.release()
        self.d_heuristics.release()
        self.ctx.release()
