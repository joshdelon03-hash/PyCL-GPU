import multiprocessing
import queue
import time
import traceback
import threading
import os

# We need a pickleable container for tasks
class TaskPayload:
    def __init__(self, task_id, target, args, kwargs):
        self.task_id = task_id
        self.target = target
        self.args = args
        self.kwargs = kwargs

# Alias for API compatibility with the ATP model
Task = TaskPayload

class TaskResult:
    def __init__(self, task_id, success, data):
        self.task_id = task_id
        self.success = success # True/False
        self.data = data # Result or Exception

class WorkerProcess(multiprocessing.Process):
    def __init__(self, task_queue, result_queue, max_tasks=None):
        super().__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.max_tasks = max_tasks
        self.tasks_completed = 0
        self.daemon = True # Kill when main process exits

    def run(self):
        while True:
            try:
                # 1. Get a task
                payload = self.task_queue.get()
                if payload is None: # Sentinel to stop
                    break
                
                # 2. Execute
                try:
                    result = payload.target(*payload.args, **payload.kwargs)
                    self.result_queue.put(TaskResult(payload.task_id, True, result))
                except Exception as e:
                    # Capture exception to send back
                    self.result_queue.put(TaskResult(payload.task_id, False, e))
                
                self.tasks_completed += 1
                if self.max_tasks and self.tasks_completed >= self.max_tasks:
                    # Exit to prevent memory creep
                    break
                    
            except Exception:
                # If something goes wrong with the queue itself
                traceback.print_exc()
                break

class MultiprocessingPool:
    def __init__(self, min_workers=2, max_tasks_per_worker=100):
        self.min_workers = min_workers
        self.max_tasks_per_worker = max_tasks_per_worker
        self.workers = []
        # Use multiprocessing queues
        self.task_queue = multiprocessing.Queue()
        self.result_queue = multiprocessing.Queue()
        
        # Mapping task_id -> (success_callback, failure_callback)
        self.callbacks = {}
        self.task_counter = 0
        self.lock = threading.Lock() # Protect callbacks dict and counter
        self._shutdown = False

        # Start initial workers
        for _ in range(min_workers):
            self._add_worker()

        # Start the result handler thread (runs in main process)
        self.result_handler = threading.Thread(target=self._handle_results)
        self.result_handler.daemon = True
        self.result_handler.start()

        # Start adaptive monitor (runs in main process)
        self.monitor_thread = threading.Thread(target=self._monitor_load)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def _add_worker(self):
        w = WorkerProcess(self.task_queue, self.result_queue, max_tasks=self.max_tasks_per_worker)
        w.start()
        self.workers.append(w)

    def _monitor_load(self):
        while not self._shutdown:
            time.sleep(0.5)
            try:
                with self.lock:
                    # Filter out dead workers
                    self.workers = [w for w in self.workers if w.is_alive()]
                    n_workers = len(self.workers)

                # Ensure minimum worker count (Fault Tolerance)
                while n_workers < self.min_workers:
                    self._add_worker()
                    n_workers += 1

                # Basic scaling logic
                try:
                    q_size = self.task_queue.qsize()
                except NotImplementedError:
                    q_size = 0 # Fallback logic for platforms without qsize()

                if q_size > n_workers:
                    self._add_worker()
            except Exception:
                pass

    def _handle_results(self):
        """
        Reads results from the multiprocessing queue and triggers
        callbacks in the main process context.
        """
        while not self._shutdown:
            try:
                # Use a timeout so we can check self._shutdown
                try:
                    result_packet = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                with self.lock:
                    callbacks = self.callbacks.pop(result_packet.task_id, None)
                
                if callbacks:
                    success_cb, failure_cb = callbacks
                    if result_packet.success:
                        if success_cb:
                            success_cb(result_packet.data)
                    else:
                        if failure_cb:
                            failure_cb(result_packet.data)
                        else:
                            print(f"Task failed (no callback): {result_packet.data}")

            except Exception:
                traceback.print_exc()

    def apply_async(self, target, args=(), kwargs=None, success=None, failure=None):
        if kwargs is None: kwargs = {}
        
        with self.lock:
            if self._shutdown:
                raise RuntimeError("Cannot apply_async to a shutdown pool.")
            task_id = self.task_counter
            self.task_counter += 1
            self.callbacks[task_id] = (success, failure)
        
        payload = TaskPayload(task_id, target, args, kwargs)
        self.task_queue.put(payload)

    def join(self):
        """
        Wait for all tasks to be processed.
        """
        while True:
            with self.lock:
                # Use qsize() with care, or just check callbacks
                if len(self.callbacks) == 0:
                    break
            time.sleep(0.1)

    def shutdown(self):
        """
        Stops all workers and threads cleanly.
        """
        with self.lock:
            self._shutdown = True
        
        # 1. Send None sentinels to all workers
        for _ in range(len(self.workers)):
            self.task_queue.put(None)
        
        # 2. Wait for workers to finish
        for w in self.workers:
            w.join(timeout=2.0)
            if w.is_alive():
                w.terminate()
        
        # 3. Clear queues to free up memory
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        print("AMP Pool shutdown complete.")

# Singleton
_global_mp_pool = None

def GlobalMPPool(min_workers=None, max_tasks_per_worker=100):
    global _global_mp_pool
    if _global_mp_pool is None or _global_mp_pool._shutdown:
        if min_workers is None:
            min_workers = os.cpu_count() or 2
        _global_mp_pool = MultiprocessingPool(min_workers, max_tasks_per_worker)
    return _global_mp_pool

def async_call(target, *args, **kwargs):
    success = kwargs.pop('success', None)
    failure = kwargs.pop('failure', None)
    pool = GlobalMPPool()
    pool.apply_async(target, args=args, kwargs=kwargs, success=success, failure=failure)
