import multiprocessing
import queue
import time
import traceback
import threading
import os

import os
import sys

try:
    from framework.neural import LatencyPredictor
except ImportError:
    LatencyPredictor = None

try:
    from framework.algo.astar import AStarRouter
except ImportError:
    AStarRouter = None

# We need a pickleable container for tasks
class TaskPayload:
    def __init__(self, task_id, target, args, kwargs, metadata=None):
        self.task_id = task_id
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.metadata = metadata or {} # For storing data_size, target_id, etc.

# Alias for API compatibility with the ATP model
Task = TaskPayload

class TaskResult:
    def __init__(self, task_id, success, data, duration=0.0):
        self.task_id = task_id
        self.success = success # True/False
        self.data = data # Result or Exception
        self.duration = duration # Time taken in seconds

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
                
                # 2. Execute with timing
                start_time = time.perf_counter()
                try:
                    result = payload.target(*payload.args, **payload.kwargs)
                    duration = time.perf_counter() - start_time
                    self.result_queue.put(TaskResult(payload.task_id, True, result, duration))
                except Exception as e:
                    duration = time.perf_counter() - start_time
                    # Capture exception to send back
                    self.result_queue.put(TaskResult(payload.task_id, False, e, duration))
                
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
        
        # Mapping task_id -> (success_callback, failure_callback, metadata)
        self.callbacks = {}
        self.task_counter = 0
        self.lock = threading.Lock() # Protect callbacks dict and counter
        self._shutdown = False

        # Neural Predictor
        self.predictor = LatencyPredictor() if LatencyPredictor else None
        
        # Day 4: A* Router
        self.router = AStarRouter(width=16, height=16) if AStarRouter else None

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
        """
        Neural-driven dynamic scaling. Uses predicted latency pressure 
        to decide whether to spawn or reap workers.
        """
        PRESSURE_THRESHOLD = 0.1 # 100ms of predicted work justifies a new worker
        
        while not self._shutdown:
            time.sleep(0.5)
            try:
                with self.lock:
                    # 1. Clean up dead workers
                    self.workers = [w for w in self.workers if w.is_alive()]
                    n_workers = len(self.workers)
                    max_workers = os.cpu_count() or 4

                    # 2. Calculate "Queue Pressure" using Neural Predictions
                    # (We use the most recent predictions stored in the callback metadata)
                    total_pressure = 0.0
                    if self.predictor:
                        for task_id in list(self.callbacks.keys()):
                            cb_data = self.callbacks.get(task_id)
                            if cb_data:
                                # We can't easily re-predict without args, so we use a 
                                # weighted average of recent latencies as a fallback
                                # or just count tasks if the model isn't ready.
                                total_pressure += 0.01 # Placeholder base pressure
                    
                    # 3. Decision Logic
                    # Ensure minimum worker count
                    if n_workers < self.min_workers:
                        self._add_worker()
                    
                    # Scale Up: If pressure is high and we have headroom
                    elif n_workers < max_workers:
                        # Use qsize as a proxy for raw pressure if predictor is warming up
                        try:
                            q_size = self.task_queue.qsize()
                        except:
                            q_size = 0
                            
                        if q_size > n_workers * 2: # High raw volume
                            self._add_worker()
                        elif total_pressure > PRESSURE_THRESHOLD * n_workers:
                            self._add_worker()

                    # Scale Down: If idle (Optional, could be added for extreme robustness)
                    
            except Exception as e:
                # print(f"Monitor Error: {e}")
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
                    callback_data = self.callbacks.pop(result_packet.task_id, None)
                
                if callback_data:
                    success_cb, failure_cb, metadata = callback_data
                    
                    # Update neural predictor if available
                    if self.predictor:
                        self.predictor.update(
                            metadata['data_size'],
                            metadata['queue_depth'],
                            metadata['worker_count'],
                            metadata['target_id'],
                            result_packet.duration
                        )

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

    def _estimate_data_size(self, args, kwargs):
        size = 0
        # Basic heuristic for numpy/torch/list/dict
        for a in list(args) + list(kwargs.values()):
            try:
                if hasattr(a, 'nbytes'):
                    size += a.nbytes
                elif hasattr(a, 'element_size'):
                    size += a.element_size() * a.nelement()
                else:
                    size += sys.getsizeof(a)
            except:
                pass
        return size

    def apply_async(self, target, args=(), kwargs=None, success=None, failure=None):
        if kwargs is None: kwargs = {}
        
        data_size = self._estimate_data_size(args, kwargs)
        target_id = hash(target) % 10000
        
        with self.lock:
            if self._shutdown:
                raise RuntimeError("Cannot apply_async to a shutdown pool.")
            task_id = self.task_counter
            self.task_counter += 1
            
            queue_depth = self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
            worker_count = len([w for w in self.workers if w.is_alive()])
            
            metadata = {
                'data_size': data_size,
                'queue_depth': queue_depth,
                'worker_count': worker_count,
                'target_id': target_id
            }
            
            # Predict latency if possible
            if self.predictor:
                prediction = self.predictor.predict(data_size, queue_depth, worker_count, target_id)
                # print(f"Neural Prediction for Task {task_id}: {prediction:.6f}s")
            
            self.callbacks[task_id] = (success, failure, metadata)
        
        payload = TaskPayload(task_id, target, args, kwargs, metadata=metadata)
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

    def save_checkpoint(self, filename="amp_checkpoint.npy"):
        """
        Day 4 Recovery Protocol: Saves the current congestion map and 
        task metadata to a .npy file to protect against hardware failure.
        """
        if not self.router:
            return
            
        try:
            state = {
                'congestion': self.router.congestion,
                'task_counter': self.task_counter,
                'timestamp': time.time()
            }
            np.save(filename, state)
            # print(f"Checkpoint saved: {filename}")
        except Exception as e:
            # print(f"Checkpoint Error: {e}")
            pass

    def shutdown(self):
        """
        Stops all workers and threads cleanly.
        """
        self.save_checkpoint() # Day 4: Save before exit
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
