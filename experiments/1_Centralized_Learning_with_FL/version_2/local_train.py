import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from FL_model import LSTMTabular
from dataset import load_bank_dataset

BANK_ID = os.environ["BANK_ID"]
LOCAL_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 256
HIDDEN_DIM = 30
NUM_LAYERS = 1
L2_NORM_CLIP = 1.0
NOISE_MULTIPLIER = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_bank_dataset(BANK_ID, data_path="/data")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

input_dim = dataset.tensors[0].shape[1]

# Load global model
global_model = LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
global_model.load_state_dict(torch.load("/logs/global_model.pt"))

# Weighted loss
all_labels = dataset.tensors[1]
pos = (all_labels == 1).sum().item()
neg = (all_labels == 0).sum().item()
weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

# Local training with DP
global_model.train()
optimizer = optim.Adam(global_model.parameters(), lr=LR)

for _ in range(LOCAL_EPOCHS):
    for x, y in train_loader:
        x, y = x.unsqueeze(1).to(device), y.to(device)  # Add time dim
        optimizer.zero_grad()
        loss = criterion(global_model(x), y)
        loss.backward()
        
        # DP
        torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=L2_NORM_CLIP)
        for p in global_model.parameters():
            if p.grad is not None:
                noise_tensor = torch.randn_like(p.grad) * (L2_NORM_CLIP * NOISE_MULTIPLIER)
                p.grad.add_(noise_tensor)
        
        optimizer.step()

# Save local model
torch.save(global_model.state_dict(), f"/logs/{BANK_ID}_local_model.pt")