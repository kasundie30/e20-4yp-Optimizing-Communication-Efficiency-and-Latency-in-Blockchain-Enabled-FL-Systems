"""LocalClient — fog node (bank) that trains the global model locally."""
import time, io, copy
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.lstm_model import LSTMTabular
import config

class LocalClient:
    def __init__(self, client_id, X_train, y_train):
        self.client_id = client_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X_train, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        self.dataset = TensorDataset(X_t, y_t)
        self.n_samples = len(X_t)
        pos = float((y_train == 1).sum())
        neg = float((y_train == 0).sum())
        self.pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32,
                                        device=self.device)

    def train_one_round(self, global_state_dict):
        """Train locally for LOCAL_EPOCHS. Return (state_dict, n_samples, secs, bytes)."""
        model = LSTMTabular(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS).to(self.device)
        model.load_state_dict(copy.deepcopy(global_state_dict))
        model.train()
        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        loader = DataLoader(self.dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        t0 = time.perf_counter()
        for _ in range(config.LOCAL_EPOCHS):
            for xb, yb in loader:
                xb = xb.unsqueeze(1).to(self.device)
                yb = yb.unsqueeze(1).to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
        elapsed = time.perf_counter() - t0
        buf = io.BytesIO(); torch.save(model.state_dict(), buf)
        local_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        return local_sd, self.n_samples, elapsed, buf.tell()
