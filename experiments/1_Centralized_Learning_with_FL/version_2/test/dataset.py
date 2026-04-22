import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def load_bank_dataset(bank_id):
    """
    Loads dataset for a single bank
    Path: /home/fyp-group-18/FYP-Group18/data/processed/2_bank_silos/bank_X/local_data.csv
    """
    path = f"/home/fyp-group-18/FYP-Group18/data/processed/2_bank_silos/{bank_id}/local_data.csv"

    df = pd.read_csv(path)

    # last column is label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Feature scaling (local)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return TensorDataset(X_tensor, y_tensor)
