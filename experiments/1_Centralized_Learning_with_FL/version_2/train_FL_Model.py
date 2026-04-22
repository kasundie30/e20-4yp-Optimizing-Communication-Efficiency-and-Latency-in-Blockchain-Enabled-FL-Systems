# Author: Wethmi
# train_FL_Model.py

# This script is used to train the global model using the FL framework.

import os
import time
import subprocess
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, f1_score
from FL_model import LSTMTabular
from dataset import load_bank_dataset

# Hyperparams
NUM_CLIENTS = 10 # Number of banks participating
NUM_ROUNDS = 10 # Number of communication rounds
LOCAL_EPOCHS = 2 # Epochs trained inside each bank
LR = 1e-3 # Learning rate
BATCH_SIZE = 256 # Local training batch size
HIDDEN_DIM = 30 # LSTM hidden dimension
NUM_LAYERS = 1 # LSTM num layers, depth
L2_NORM_CLIP = 1.0 # DP clipping norm (DP-related) gradient clipping
NOISE_MULTIPLIER = 0.05 # DP noise multiplier (DP-related) noise scale

# Uses GPU if available Otherwise falls back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining FL clients
banks = [f"bank_{i}" for i in range(1, 11)]

# Get input_dim from one dataset
sample_dataset = load_bank_dataset("bank_1")
input_dim = sample_dataset.tensors[0].shape[1]

# Build the Docker image
project_dir = "/home/fyp-group-18/FYP-Group18"
subprocess.run(["docker", "build", "-f", "docker/Dockerfile.bank", "-t", "bank_fl", "."], cwd=project_dir, check=True)

# Initialize global model
global_model = LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
torch.save(global_model.state_dict(), os.path.join(project_dir, "logs", "global_model.pt"))

communication_overhead = 0
start_time = time.time()
f1_history = []

# Federated Learning loop
for r in range(NUM_ROUNDS):
    print(f"Starting round {r+1}")
    for bank in banks:
        # Run container for local training
        cmd = [
            "docker", "run", "--rm",
            "-e", f"BANK_ID={bank}",
            "-v", f"{project_dir}/data/processed/3_local_silo_balancing:/data",
            "-v", f"{project_dir}/logs:/logs",
            "bank_fl"
        ]
        subprocess.run(cmd, cwd=project_dir, check=True)
    
    # Load local models and aggregate
    local_models = []
    weights = []
    for bank in banks:
        model = LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(os.path.join(project_dir, "logs", f"{bank}_local_model.pt")))
        local_models.append(model)
        # Assume equal weights for simplicity
        weights.append(1.0)
    
    # FedAvg aggregation
    total_w = sum(weights)
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.stack([m.state_dict()[key].float() * (w/total_w) for m, w in zip(local_models, weights)], 0).sum(0)
    global_model.load_state_dict(global_dict)
    # Save updated global model
    torch.save(global_model.state_dict(), os.path.join(project_dir, "logs", "global_model.pt"))
    
    # Communication overhead calculation
    num_params = sum(p.numel() for p in global_model.parameters())
    communication_overhead += num_params * 4 * NUM_CLIENTS
    
    # Eval on bank_1 data (Uses one bank’s data as validation)
    # due to lack of global test data
    eval_dataset = load_bank_dataset("bank_1")
    all_X = eval_dataset.tensors[0]
    all_y = eval_dataset.tensors[1].numpy().flatten()
    probs = torch.sigmoid(global_model(all_X.unsqueeze(1).to(device))).detach().cpu().numpy().flatten()
    preds = (probs >= 0.5).astype(int)
    f1 = f1_score(all_y, preds)
    f1_history.append(f1)
    print(f"Round {r+1} | F1: {f1:.4f}")

end_time = time.time()
# End-to-end latency
end_to_end_latency = end_time - start_time

# Convergence
max_f1 = max(f1_history)
convergence_round = next((i+1 for i, f in enumerate(f1_history) if f >= 0.95 * max_f1), NUM_ROUNDS)

# Model quality
precision, recall, f1_final, _ = precision_recall_fscore_support(all_y, preds, average='binary')

print("\nEvaluation Results:")
print(f"Communication Overhead: {communication_overhead} bytes")
print(f"End-to-End Latency: {end_to_end_latency:.2f} seconds")
print(f"Convergence Speed: {convergence_round} rounds")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_final:.4f}")