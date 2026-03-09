import numpy as np
import time
from framework.general import GeneralTask
from framework.algo.dpj import RadixReorder, batch_dot_product

# Configuration
N_VECTORS = 5000       # Number of sparse vectors (U)
DENSE_ROWS = 10000     # Rows in V (e.g., vocabulary size)
DENSE_COLS = 64        # Embedding dimension
SPARSITY = 0.01        # 1% non-zero entries
PAGE_SIZE = 128        # "Page" size for reordering simulation

def generate_data():
    print(f"Generating synthetic data: {N_VECTORS} vectors, {DENSE_ROWS}x{DENSE_COLS} dense matrix...")
    
    # 1. Create Dense Matrix V (The "Model")
    V = np.random.rand(DENSE_ROWS, DENSE_COLS).astype(np.float32)
    
    # 2. Create Sparse Vectors U (The "Training Data")
    # Represented as list of (indices, values)
    U = []
    for _ in range(N_VECTORS):
        # Randomly choose non-zero indices
        nnz = int(DENSE_ROWS * SPARSITY)
        indices = np.random.choice(DENSE_ROWS, nnz, replace=False)
        indices.sort() # Sorting helps cache slightly, but order matters for DPJ
        values = np.random.rand(nnz).astype(np.float32)
        U.append((indices, values))
        
    return U, V

def run_benchmark(name, order, U, V, batch_size=500):
    print(f"\n--- Running {name} ---")
    
    # 1. Initialize Generic AMP Task
    # We pass a dummy function because we'll use run_async directly
    task = GeneralTask(func=batch_dot_product)
    
    # 2. Split into batches
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    print(f"Processing {len(batches)} batches of size {batch_size}...")
    
    start_time = time.time()
    
    # 3. Submit batches
    for batch_indices in batches:
        # NOTE: In a real distributed system, we wouldn't pass U and V 
        # in every call. We'd use shared memory or a database.
        # For this demo, pickling overhead might mask the algo speedup,
        # but we are proving the *orchestration* capability.
        
        # To avoid massive pickling, we could pass just the indices if 
        # U and V were in shared memory, but let's just pass them for now.
        task.run_async(batch_indices, U, V)
        
    # 4. Wait
    task.wait_all()
    duration = time.time() - start_time
    
    print(f"Completed in {duration:.4f}s")
    task.shutdown()
    return duration

def main():
    print("=== PyCL-GPU: Dot Product Join (DPJ) Demo ===")
    print("Demonstrating 'General Task' AMP execution + Radix Reordering Algo")
    
    U, V = generate_data()
    
    # --- Baseline: Random Order ---
    random_order = list(range(N_VECTORS))
    np.random.shuffle(random_order)
    t_random = run_benchmark("Baseline (Random Order)", random_order, U, V)
    
    # --- Optimization: Radix Reordering ---
    print("\n[Algo] Computing Radix Reordering...")
    reorderer = RadixReorder(page_size=PAGE_SIZE)
    t_sort_start = time.time()
    dpj_order = reorderer.reorder(U, V.shape)
    t_sort = time.time() - t_sort_start
    print(f"Reordering took {t_sort:.4f}s")
    
    t_dpj = run_benchmark("Dot Product Join (Radix Order)", dpj_order, U, V)
    
    print(f"\nResults:")
    print(f"Baseline: {t_random:.4f}s")
    print(f"DPJ Order: {t_dpj:.4f}s")
    if t_dpj < t_random:
        print(f"Speedup: {t_random / t_dpj:.2f}x (excluding sort time)")
        print("Note: In-memory simulation speedup is limited by Python overhead.")
        print("Real-world 'Big Model' speedups (disk/GPU) are typically 10x-100x.")
    else:
        print("Note: Python pickling overhead likely dominated the cache benefits in this small demo.")

if __name__ == "__main__":
    main()
