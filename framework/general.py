import amp
import time

class GeneralTask:
    """
    A wrapper for executing arbitrary Python functions using the 
    AMP (Adaptive Multiprocessing Pool).
    
    This transforms AMP from a "GPU-only" tool into a universal 
    accelerator for PyTorch, Scikit-Learn, or raw Algorithm logic.
    """
    def __init__(self, func, pool_size=None):
        """
        Args:
            func: The target function to execute in parallel.
            pool_size: Optional number of workers (defaults to CPU count).
        """
        self.func = func
        # Ensure the global pool is running
        amp.GlobalMPPool(min_workers=pool_size)

    def run_async(self, *args, **kwargs):
        """
        Submits the function to the pool.
        
        Args:
            success (callable): Optional callback for success.
            failure (callable): Optional callback for failure.
            *args, **kwargs: Arguments passed to your target function.
        """
        # Extract callbacks from kwargs if they exist
        success = kwargs.pop('success', None)
        failure = kwargs.pop('failure', None)
        
        # We wrap the user's function to ensure it runs cleanly in the worker
        # (The worker logic is already generic in amp.pool, so we can pass func directly)
        amp.async_call(
            self.func,
            *args, 
            success=success, 
            failure=failure, 
            **kwargs
        )

    def wait_all(self):
        """Waits for all currently submitted tasks to finish."""
        amp.GlobalMPPool().join()

    def shutdown(self):
        """Cleans up the pool resources."""
        amp.GlobalMPPool().shutdown()
