# Approach 4: Federated Learning (LSTM + Differential Privacy)
# Author : Methmi 
# Date : 29th December 2025

import os
import time
import numpy as np
import pandas as pd
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix, f1_score
)

# =========================
# 0) Setup & Config
# =========================
wandb.login()

PROJECT = "fl-fraud-compare"
GROUP = os.environ.get("WANDB_GROUP", "compare4-" + str(int(time.time())))

# DP + FL Hyperparams
NUM_CLIENTS = 10
NUM_ROUNDS = 10
LOCAL_EPOCHS = 2
LR = 1e-3
BATCH_SIZE = 256
HIDDEN_DIM = 30
NUM_LAYERS = 1
L2_NORM_CLIP = 1.0
NOISE_MULTIPLIER = 0.05
RANDOM_STATE = 42

run = wandb.init(
    project=PROJECT,
    group=GROUP,
    job_type="train",
    name="Approach4-LSTM-DP",
    config={
        "approach": "Approach4_LSTM_DP",
        "model": "LSTMTabular",
        "dp": True,
        "dp_clip": L2_NORM_CLIP,
        "dp_noise": NOISE_MULTIPLIER,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS
    }
)

# =========================
# 1) Data Loading (No SMOTE for DP)
# =========================
file_path = "/home/e20fyptemp3/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/raw/creditcard.csv"
df = pd.read_csv(file_path).drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

X = df.drop("Class", axis=1).values
y = df["Class"].values.astype(int)

X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
X_train_raw, X_val_raw, y_train_raw, y_val = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_raw)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

client_data = np.array_split(X_train_scaled, NUM_CLIENTS)
client_labels = np.array_split(y_train_raw, NUM_CLIENTS)
client_sizes = [len(cd) for cd in client_data]

# =========================
# 2) Model & DP Helpers
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_train_scaled.shape[1]

class LSTMTabular(nn.Module):
    def __init__(self, input_dim=30, hidden_dim=30, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        return self.fc(h[-1])

# Weighted loss to handle imbalance without SMOTE
pos = (y_train_raw == 1).sum()
neg = (y_train_raw == 0).sum()
weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

def local_train_private(model, cx, cy, clip, noise):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    ds = TensorDataset(torch.tensor(cx).float().unsqueeze(1), torch.tensor(cy).float().view(-1, 1))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    
    for _ in range(LOCAL_EPOCHS):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            
            # DP Step 1: Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            
            # DP Step 2: Noise addition
            for p in model.parameters():
                if p.grad is not None:
                    noise_tensor = torch.randn_like(p.grad) * (clip * noise)
                    p.grad.add_(noise_tensor)
            
            optimizer.step()
    return model

def fedavg_weighted(models, weights, in_dim):
    global_model = LSTMTabular(in_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
    total_w = sum(weights)
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([m.state_dict()[key].float() * (w/total_w) for m, w in zip(models, weights)], 0).sum(0)
    global_model.load_state_dict(global_dict)
    return global_model

@torch.no_grad()
def get_probs(model, X_np):
    model.eval()
    X_t = torch.tensor(X_np).float().unsqueeze(1).to(device)
    return torch.sigmoid(model(X_t)).cpu().numpy().flatten()

# =========================
# 3) Main Training Loop
# =========================
global_model = LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
client_models = [LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device) for _ in range(NUM_CLIENTS)]

for r in range(NUM_ROUNDS):
    for i in range(NUM_CLIENTS):
        client_models[i].load_state_dict(global_model.state_dict())
        client_models[i] = local_train_private(client_models[i], client_data[i], client_labels[i], L2_NORM_CLIP, NOISE_MULTIPLIER)
    
    global_model = fedavg_weighted(client_models, client_sizes, input_dim)
    
    # Round Eval
    val_probs = get_probs(global_model, X_val_scaled)
    val_f1 = f1_score(y_val, (val_probs >= 0.5).astype(int))
    wandb.log({"round": r+1, "val/f1": val_f1, "dp/noise": NOISE_MULTIPLIER})
    print(f"Round {r+1} | Val F1: {val_f1:.4f}")

# =========================
# 4) Threshold Tuning & Final Logging
# =========================
val_probs = get_probs(global_model, X_val_scaled)
test_probs = get_probs(global_model, X_test_scaled)

best_t, best_f1 = 0.5, -1
for t in np.linspace(0.01, 0.9, 50):
    f1 = f1_score(y_val, (val_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

test_preds = (test_probs >= best_t).astype(int)
probas_2col = np.stack([1-test_probs, test_probs], axis=1)

# Visuals
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_test, preds=test_preds, class_names=["legit", "fraud"]),
    "roc": wandb.plot.roc_curve(y_test, probas_2col),
    "pr": wandb.plot.pr_curve(y_test, probas_2col)
})

run.summary.update({
    "final/test_f1": f1_score(y_test, test_preds),
    "final/test_auc": roc_auc_score(y_test, test_probs),
    "final/test_recall": precision_recall_fscore_support(y_test, test_preds, average='binary')[1],
    "best_threshold": best_t
})

run.finish()