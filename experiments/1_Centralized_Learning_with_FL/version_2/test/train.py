import os
import time
import torch
from torch.utils.data import DataLoader
from model import FraudNet
from dataset import load_bank_dataset

BANK_ID = os.environ["BANK_ID"]

dataset = load_bank_dataset(BANK_ID)

train_loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True
)

dataset = load_bank_dataset(BANK_ID)

input_dim = dataset.tensors[0].shape[1]
model = FraudNet(input_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

start = time.time()

model.train()
for x, y in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

latency = time.time() - start

print(f"[{BANK_ID}] Training latency: {latency:.2f}s")

torch.save(
    model.state_dict(),
    f"/logs/{BANK_ID}_weights.pt"
)
