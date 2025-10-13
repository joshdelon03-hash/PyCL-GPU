import numpy as np
import time
from framework.task import ParallelTask
from framework.buffer import DeviceBuffer

def sieve_for_base_primes(limit):
    """A simple Sieve of Eratosthenes to get primes up to a small limit."""
    primes = []
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
    return np.array(primes, dtype=np.int32)

def main():
    """
    Uses a parallel segmented sieve algorithm to find prime numbers.
    """
    print("--- Massively Parallel Prime Number Counter (Sieve Algorithm) ---")

    # This kernel is very different. Each work-item takes one "base prime"
    # and marks all of its multiples in the main results array as not-prime.
    kernel_code = """
    __kernel void sieve_kernel(__global int *results,
                               __global const int *base_primes,
                               const int max_n)
    {
        int base_prime = base_primes[get_global_id(0)];
        
        // Start marking multiples from base_prime * base_prime.
        // All smaller multiples would have been marked by smaller primes.
        long start = (long)base_prime * base_prime;

        // We need to offset because the results array is 0-indexed
        // but our numbers start from 2.
        long start_index = start - 2;

        for (long i = start_index; i < max_n; i += base_prime) {
            if (i >= 0) { // Ensure we don't write to negative indices
                results[i] = 0; // Mark as not prime
            }
        }
    }
    """

    try:
        # 1. Define the total range of numbers to check.
        N = 100_000_000
        print(f"Preparing to find primes up to {N} using a parallel sieve...")

        # 2. CPU pre-calculation: Find all base primes up to sqrt(N).
        print("Step 1: CPU is finding all base primes up to sqrt(N)...")
        sqrt_n = int(np.sqrt(N))
        base_primes = sieve_for_base_primes(sqrt_n)
        print(f"Found {len(base_primes)} base primes (e.g., {base_primes[:10]}...).\n")

        # 3. Create the ParallelTask. This compiles the new kernel.
        sieve_task = ParallelTask(kernel_code)

        # 4. Create GPU buffers.
        print("Step 2: Creating GPU buffers and transferring data...")
        # Buffer for the small list of base primes (read-only for the kernel)
        base_primes_buffer = DeviceBuffer.from_numpy(sieve_task.ctx, base_primes)
        
        # The main results buffer. Initialize with all 1s (all are potentially prime).
        results_host_template = np.ones(N - 1, dtype=np.int32) # (Checking from 2 to N)
        results_buffer = DeviceBuffer.from_numpy(sieve_task.ctx, results_host_template)
        print("Buffers created.\n")

        # 5. Execute the kernel.
        # The global size is now the number of BASE PRIMES. Each thread handles one.
        global_size = (len(base_primes),)
        kernel_args = [results_buffer, base_primes_buffer, N - 1]
        
        print(f"Step 3: Launching kernel on GPU with {global_size[0]} work-items (one for each base prime)...")
        start_time = time.time()

        sieve_task.execute(global_size, kernel_args)
        
        # 6. Read the results back from the GPU.
        results_host = results_buffer.read()

        end_time = time.time()
        print(f"GPU computation finished in {end_time - start_time:.4f} seconds.\n")

        # 7. Process the results on the CPU.
        print("--- Results ---")
        # The count is the sum of the boolean array. We add 1 because our base primes list includes the first prime (2).
        prime_count = np.sum(results_host)
        print(f"Found {prime_count} prime numbers between 2 and {N}.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()