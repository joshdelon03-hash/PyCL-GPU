import numpy as np
from collections import defaultdict

class RadixReorder:
    """
    Implements the Radix Sort Reordering heuristic from the 'Dot-Product Join' paper (1602.08845).
    
    Goal: Given a set of sparse vectors U (which access pages of dense matrix V),
    reorder U so that vectors accessing similar pages are grouped together.
    This minimizes 'page misses' when processing V.
    """
    def __init__(self, page_size=4096):
        self.page_size = page_size

    def compute_page_frequency(self, sparse_vectors, V_shape):
        """
        Step 1: Compute how frequently each page of V is accessed.
        
        Args:
            sparse_vectors: List of (indices, values) tuples for each vector in U.
            V_shape: Shape of the dense matrix V (rows, cols).
        
        Returns:
            A dictionary mapping page_id -> frequency.
        """
        page_freq = defaultdict(int)
        
        for indices, _ in sparse_vectors:
            # Which pages does this vector touch?
            pages_touched = set()
            for idx in indices:
                page_id = idx // self.page_size
                pages_touched.add(page_id)
            
            for p in pages_touched:
                page_freq[p] += 1
                
        return page_freq

    def reorder(self, sparse_vectors, V_shape):
        """
        Step 2 & 3: Reorder the vectors based on page frequency using a Radix-like approach.
        
        The paper suggests:
        1. Identify the 'Most Frequent Page' across all vectors.
        2. Group vectors that access this page together.
        3. Within those groups, recurse or sort by the next most frequent page.
        
        Simplified Implementation (Bucket Sort by Top Page):
        We bucket vectors by their *most frequent* accessed page.
        """
        # 1. Get global page frequencies
        global_page_freq = self.compute_page_frequency(sparse_vectors, V_shape)
        
        # 2. Assign each vector to a 'primary bucket' based on its HOTTEST page
        buckets = defaultdict(list)
        
        for i, (indices, values) in enumerate(sparse_vectors):
            if len(indices) == 0:
                buckets[-1].append(i) # Empty vector
                continue
                
            # Find the most frequent page this vector accesses
            best_page = -1
            max_freq = -1
            
            for idx in indices:
                page_id = idx // self.page_size
                freq = global_page_freq[page_id]
                if freq > max_freq:
                    max_freq = freq
                    best_page = page_id
            
            # Tie-breaking: if frequencies equal, pick smaller page_id (or random)
            buckets[best_page].append(i)
            
        # 3. Flatten buckets to get the new order
        # We process buckets in order of page frequency (most popular pages first)
        sorted_pages = sorted(global_page_freq.keys(), key=lambda p: global_page_freq[p], reverse=True)
        
        new_order = []
        seen_vectors = set()
        
        # Add vectors from the hottest buckets first
        for page_id in sorted_pages:
            if page_id in buckets:
                for vec_idx in buckets[page_id]:
                    if vec_idx not in seen_vectors:
                        new_order.append(vec_idx)
                        seen_vectors.add(vec_idx)
        
        # Add any remaining (e.g., from pages not in the top list or empty vectors)
        for page_id in buckets:
            if page_id not in global_page_freq: # Should be covered, but for safety
                 for vec_idx in buckets[page_id]:
                    if vec_idx not in seen_vectors:
                        new_order.append(vec_idx)
                        seen_vectors.add(vec_idx)

        return new_order

def batch_dot_product(batch_indices, sparse_U, dense_V):
    """
    A function to be run by AMP workers.
    Computes dot product for a batch of vectors.
    """
    results = {}
    # We assume sparse_U is passed fully or shared via memory mapping in a real DB
    # For this demo, we pass the small batch data or assume it's available.
    
    # In a real "Big Model" scenario, we would only load the needed pages of V here.
    # That's the benefit of the reordering!
    
    for original_idx in batch_indices:
        u_indices, u_values = sparse_U[original_idx]
        
        # Dot product: sum(u_val * V[u_idx])
        # Since V is dense (N x D), and u is 1 x N (sparse), result is 1 x D
        
        # Simple dense dot for the demo
        # res = np.zeros(dense_V.shape[1])
        # for i, idx in enumerate(u_indices):
        #     res += u_values[i] * dense_V[idx]
            
        # Optimization: use numpy advanced indexing
        if len(u_indices) > 0:
            rows = dense_V[u_indices] # Gather rows
            # u_values is (nz, ), rows is (nz, D) -> weighted sum
            # broadcast u_values to (nz, 1)
            weighted = rows * u_values[:, np.newaxis]
            res = np.sum(weighted, axis=0)
        else:
            res = np.zeros(dense_V.shape[1])
            
        results[original_idx] = res
        
    return results
