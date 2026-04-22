# Approach 2: Centralized Learning with FL framework (MLP + SMOTE + StandardScaler) 
# Author : Methmi 
# Date : 29th December 2025

import wandb
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

# =========================
# 0) Setup & W&B
# =========================
wandb.login()
PROJECT = "fl-fraud-compare"
GROUP = "compare4-" + str(int(time.time()))
print("W&B GROUP =", GROUP)

run = wandb.init(
    project=PROJECT,
    group=GROUP,
    job_type="train",
    name="Approach2-MLP",
    config={
        "approach": "Approach2_MLP",
        "model": "MLP",
        "dp": False,
        "num_rounds": 15,
        "local_epochs": 2,
        "lr": 1e-3,
        "batch_size": 256,
        "weighted_fedavg": True,
        "weight_decay": 1e-5,
        "criterion": "BCEWithLogitsLoss"
    }
)

# =========================
# 1) Load + Preprocess
# =========================
file_path = "/home/e20fyptemp3/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/raw/creditcard.csv"
df = pd.read_csv(file_path).drop_duplicates()
df.fillna(df.mean(numeric_only=True), inplace=True)

X = df.drop("Class", axis=1).values
y = df["Class"].values

# Stratified splits
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# SMOTE
smote = SMOTE(sampling_strategy="minority", random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_raw)

# Split for clients
num_clients = 10
client_data = np.array_split(X_train_res, num_clients)
client_labels = np.array_split(y_train_res, num_clients)
client_sizes = [len(cd) for cd in client_data]

# =========================
# 2) Model & FL Helpers
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = X_train_res.shape[1]

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1) # Logits
        )

    def forward(self, x):
        return self.net(x)

def make_loss_with_pos_weight(y_binary):
    pos = (y_binary == 1).sum()
    neg = (y_binary == 0).sum()
    weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight)

criterion = make_loss_with_pos_weight(y_train_res)

def fedavg_weighted(models, weights, in_dim):
    global_model = MLP(in_dim).to(device)
    total_weight = float(sum(weights))
    
    # State dict aggregation (better for BatchNorm tracking)
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([m.state_dict()[key].float() * (w / total_weight) for m, w in zip(models, weights)], 0).sum(0)
    
    global_model.load_state_dict(global_dict)
    return global_model

def local_train(model, cx, cy, epochs=2, lr=1e-3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = TensorDataset(torch.tensor(cx).float(), torch.tensor(cy).float().view(-1, 1))
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
    return model

@torch.no_grad()
def evaluate_global(model, X_np, y_np):
    model.eval()
    X_t = torch.tensor(X_np).float().to(device)
    y_t = torch.tensor(y_np).float().view(-1, 1).to(device)
    
    logits = model(X_t)
    loss = criterion(logits, y_t).item()
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    acc = accuracy_score(y_np, (probs >= 0.5).astype(int))
    return loss, acc, probs

# =========================
# 3) FL Training Loop
# =========================
global_model = MLP(in_dim).to(device)
models = [MLP(in_dim).to(device) for _ in range(num_clients)]

for r in range(run.config["num_rounds"]):
    # Sync & Train
    for i in range(num_clients):
        models[i].load_state_dict(global_model.state_dict())
        models[i] = local_train(models[i], client_data[i], client_labels[i], 
                                epochs=run.config["local_epochs"], lr=run.config["lr"])
    
    # Aggregate
    global_model = fedavg_weighted(models, client_sizes, in_dim)
    
    # Eval on Test Scaled (per round tracking)
    g_loss, g_acc, g_probs = evaluate_global(global_model, X_test_scaled, y_test)
    
    auc = roc_auc_score(y_test, g_probs)
    wandb.log({
        "round": r + 1,
        "global/loss": g_loss,
        "global/acc": g_acc,
        "global/auc": auc
    }, step=r)
    print(f"Round {r+1} | Loss: {g_loss:.4f} | Acc: {g_acc:.4f} | AUC: {auc:.4f}")

# =========================
# 4) Threshold Tuning & Final Evaluation
# =========================
_, _, val_probs = evaluate_global(global_model, X_val_scaled, y_val_raw)
thresholds = np.linspace(0.01, 0.99, 99)
best_t = 0.5
best_f1 = -1

for t in thresholds:
    f1 = f1_score(y_val_raw, (val_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"\nBest threshold: {best_t:.2f} (Val F1: {best_f1:.4f})")

# Final Test Metrics
test_loss, _, test_probs = evaluate_global(global_model, X_test_scaled, y_test)
test_preds = (test_probs >= best_t).astype(int)

# Log Visuals
probas_2col = np.stack([1 - test_probs, test_probs], axis=1)
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(y_true=y_test, preds=test_preds, class_names=["legit", "fraud"]),
    "roc": wandb.plot.roc_curve(y_test, probas_2col),
    "pr": wandb.plot.pr_curve(y_test, probas_2col)
})

run.summary.update({
    "final/test_f1": f1_score(y_test, test_preds),
    "final/test_precision": precision_score(y_test, test_preds),
    "final/test_recall": recall_score(y_test, test_preds),
    "final/test_auc": roc_auc_score(y_test, test_probs),
    "final/best_threshold": best_t
})

run.finish()