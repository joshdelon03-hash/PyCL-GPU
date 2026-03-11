import numpy as np
import time
import os
from framework.algo.astar import AStarRouter
import matplotlib.pyplot as plt

def main():
    print("--- Day 4: A* PCIe Data Routing Demo ---")
    
    # 1. Initialize Router (16x16 Grid of "Lanes")
    width, height = 16, 16
    router = AStarRouter(width=width, height=height)
    
    # 2. Simulate "Bus Noise" (Congestion)
    # We'll create some random "saturated" lanes (obstacles)
    congestion = np.random.rand(width * height).astype(np.float32)
    # Thresholding: Some areas are extremely busy (obstacles)
    # Let's make a vertical barrier
    for y in range(4, 12):
        congestion[y * width + 8] = 0.95 
    
    router.update_congestion(congestion)
    
    # 3. Define Start and End Points for a "Data Packet"
    start = (2, 2)
    end = (14, 14)
    
    print(f"Routing data from {start} to {end}...")
    
    # 4. Find Path using GPU-accelerated A*
    start_time = time.time()
    path = router.find_path(start, end)
    duration = time.time() - start_time
    
    if path:
        print(f"Path found in {duration:.4f}s! Length: {len(path)}")
        # print("Path:", path)
    else:
        print("No path found (Bus too congested).")

    # 5. Visualization (to prove it works)
    grid = congestion.reshape((height, width))
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Congestion (0.0 - 1.0)')
    
    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, marker='o', color='cyan', label='Data Route')
    
    plt.scatter([start[0]], [start[1]], color='green', s=100, label='Source')
    plt.scatter([end[0]], [end[1]], color='blue', s=100, label='Target')
    
    plt.title(f"Day 4: A* PCIe Routing Logic (GPU Heuristic)\nTime: {duration:.4f}s")
    plt.legend()
    
    # Save the result as per Day 4 goals
    plt.savefig("astar_routing_result.png")
    print("Result saved to 'astar_routing_result.png'")
    
    # router.release() # Cleanup

if __name__ == "__main__":
    main()
