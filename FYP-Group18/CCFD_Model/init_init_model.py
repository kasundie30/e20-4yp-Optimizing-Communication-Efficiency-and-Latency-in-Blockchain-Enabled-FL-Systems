# init_init_model.py
# One-time: creates the initial global model


import os
import torch
from FL_model import LSTMTabular
from dataset import load_bank_dataset

# Pick ANY existing BANK_ID folder you already have under /data
# Example: bank_1  OR  brand_1_branch_0 (depending on your dataset structure)
BANK_ID = os.environ.get("BANK_ID", "brand_1_branch_0")

# Load dataset just to get input_dim
ds = load_bank_dataset(BANK_ID, data_path="data/processed/3_local_silo_balancing")
input_dim = ds.tensors[0].shape[1]

HIDDEN_DIM = 30
NUM_LAYERS = 1

model = LSTMTabular(input_dim, HIDDEN_DIM, NUM_LAYERS)
os.makedirs("init", exist_ok=True)

torch.save(model.state_dict(), "init/global_model.pt")
print("Saved init/global_model.pt with input_dim =", input_dim)