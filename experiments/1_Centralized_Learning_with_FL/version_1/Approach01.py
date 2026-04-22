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
    job_type="preprocess+train",
    name="Approach1-MLP",
    config={
        "approach": "Approach1_MLP",
        "model": "MLP",
        "num_clients": 10,
        "num_rounds": 5,
        "local_epochs": 1,
        "lr": 1e-3,
        "batch_size": 32,
        "smote": True,
        "scaler": "StandardScaler"
    }
)

# =========================
# 1) Data Loading & Preprocessing
# =========================
file_path = "/home/e20fyptemp3/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/raw/creditcard.csv"
df = pd.read_csv(file_path).drop_duplicates()
df.fillna(df.mean(), inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# Stratified splits
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# SMOTE (Training only)
smote = SMOTE(sampling_strategy="minority", random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_raw)

# Split for clients
num_clients = 10
client_data = np.array_split(X_train_res, num_clients)
client_labels = np.array_split(y_train_res, num_clients)

# =========================
# 2) Model & FL Logic
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = X_train_res.shape[1]

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

def federated_averaging(client_models, global_model):
    state_dict_avg = global_model.state_dict()
    for key in state_dict_avg.keys():
        state_dict_avg[key] = torch.stack([m.state_dict()[key] for m in client_models], 0).mean(0)
    global_model.load_state_dict(state_dict_avg)
    return global_model

def local_train(model, data, labels, lr=0.001, epochs=1):
    model.train()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Handle both Series and Numpy
    labels_np = labels.values if hasattr(labels, "values") else labels
    dataset = TensorDataset(torch.tensor(data).float(), torch.tensor(labels_np).float().view(-1, 1))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    return model

@torch.no_grad()
def evaluate_global(model, X_np, y_np):
    model.eval()
    model.to(device)
    criterion = nn.BCELoss()
    
    X_ts = torch.tensor(X_np).float().to(device)
    y_ts = torch.tensor(y_np.values if hasattr(y_np, "values") else y_np).float().view(-1, 1).to(device)
    
    probs = model(X_ts)
    loss = criterion(probs, y_ts).item()
    preds = (probs >= 0.5).float()
    acc = (preds.eq(y_ts)).float().mean().item()
    
    return loss, acc, probs.cpu().numpy().flatten()

# =========================
# 3) Main FL Training Loop
# =========================
global_model = MLP(in_dim).to(device)

for r in range(run.config["num_rounds"]):
    client_models = []
    for i in range(num_clients):
        # 1. Sync: Copy global model to client
        local_m = MLP(in_dim).to(device)
        local_m.load_state_dict(global_model.state_dict())
        
        # 2. Local Training
        local_m = local_train(local_m, client_data[i], client_labels[i], lr=run.config["lr"])
        client_models.append(local_m)
    
    # 3. Aggregation
    global_model = federated_averaging(client_models, global_model)
    
    # 4. Evaluation
    val_loss, val_acc, _ = evaluate_global(global_model, X_val_scaled, y_val_raw)
    wandb.log({"round": r+1, "val/loss": val_loss, "val/acc": val_acc})
    print(f"Round {r+1} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# =========================
# 4) Threshold Tuning & Final Metrics
# =========================
# Get validation probabilities for tuning
_, _, val_probs = evaluate_global(global_model, X_val_scaled, y_val_raw)
thresholds = np.linspace(0.01, 0.9, 50)
best_t, best_f1 = 0.5, -1.0

for t in thresholds:
    f1 = f1_score(y_val_raw, (val_probs >= t).astype(int))
    if f1 > best_f1:
        best_f1, best_t = f1, t

# Final Test
test_loss, _, test_probs = evaluate_global(global_model, X_test_scaled, y_test)
test_preds = (test_probs >= best_t).astype(int)

# Log to W&B
probas_2col = np.stack([1 - test_probs, test_probs], axis=1)
wandb.log({
    "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test.values, preds=test_preds, class_names=["Legit", "Fraud"]),
    "roc": wandb.plot.roc_curve(y_test.values, probas_2col),
    "pr": wandb.plot.pr_curve(y_test.values, probas_2col)
})

run.summary.update({
    "test/f1": f1_score(y_test, test_preds),
    "test/precision": precision_score(y_test, test_preds),
    "test/recall": recall_score(y_test, test_preds),
    "test/auc": roc_auc_score(y_test, test_probs),
    "best_threshold": best_t
})

run.finish()