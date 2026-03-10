import torch
import numpy as np
import time
from framework.general import GeneralTask
import amp
import os

def heavy_data_task(size_mb):
    """
    Simulates a data-heavy task (e.g., processing a large tensor).
    """
    # Create a large array to simulate data transfer/memory pressure
    data = np.random.rand(size_mb * 1024 * 1024 // 8).astype(np.float64)
    # Simple compute
    res = np.sum(np.sqrt(data))
    return res

def main():
    print("--- Day 3: Neural Robustness & Stress Test ---")
    cpu_count = os.cpu_count() or 4
    print(f"System CPU Cores: {cpu_count}")
    
    # Initialize with small pool, let neural scaling take over
    task_wrapper = GeneralTask(heavy_data_task, pool_size=2)
    
    print("\nPhase 1: Warming up the Neural Model...")
    for i in range(20):
        task_wrapper.run_async(1) # 1MB tasks
    task_wrapper.wait_all()
    
    print("\nPhase 2: Stress Test (Burst of 50MB tasks)")
    print("Watching for Neural Scaling...")
    
    start_time = time.time()
    for i in range(15):
        # These tasks are 50x heavier than warm-up
        task_wrapper.run_async(50) 
    
    # Monitor worker count during execution
    while len(amp.GlobalMPPool().callbacks) > 0:
        n_workers = len([w for w in amp.GlobalMPPool().workers if w.is_alive()])
        q_size = amp.GlobalMPPool().task_queue.qsize()
        print(f"  [Monitor] Workers: {n_workers}/{cpu_count} | Queue: {q_size}", end="\r")
        time.sleep(0.5)
    
    task_wrapper.wait_all()
    duration = time.time() - start_time
    
    print(f"\n\nStress Test Complete in {duration:.2f}s")
    print("The system successfully managed burst pressure.")
    
    print("Shutting down.")
    task_wrapper.shutdown()

if __name__ == "__main__":
    main()
