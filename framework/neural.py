import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading

class LatencyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs: [data_size, queue_depth, worker_count, target_id]
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class LatencyPredictor:
    def __init__(self):
        self.model = LatencyModel()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.lock = threading.Lock()
        self.history = []
        self.batch_size = 10
        
    def predict(self, data_size, queue_depth, worker_count, target_id):
        with self.lock:
            self.model.eval()
            with torch.no_grad():
                x = torch.tensor([[float(data_size), float(queue_depth), float(worker_count), float(target_id)]], dtype=torch.float32)
                # Simple normalization (log scale for data size)
                x[0, 0] = np.log1p(x[0, 0])
                pred = self.model(x)
                return max(0.001, pred.item())

    def update(self, data_size, queue_depth, worker_count, target_id, actual_latency):
        with self.lock:
            # Store experience
            self.history.append((data_size, queue_depth, worker_count, target_id, actual_latency))
            
            # Train in small batches
            if len(self.history) >= self.batch_size:
                self._train_step()
                self.history = self.history[-100:] # Keep recent memory

    def _train_step(self):
        self.model.train()
        batch = self.history[-self.batch_size:]
        
        x_train = []
        y_train = []
        
        for ds, qd, wc, tid, al in batch:
            x = [np.log1p(float(ds)), float(qd), float(wc), float(tid)]
            x_train.append(x)
            y_train.append([float(al)])
            
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        self.optimizer.zero_grad()
        outputs = self.model(x_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
