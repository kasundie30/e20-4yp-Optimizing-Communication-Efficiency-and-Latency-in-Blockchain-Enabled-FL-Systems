import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler

def load_bank_dataset(bank_id, data_path="data/processed/3_local_silo_balancing",
                      filename_candidates=("train_ready.csv", "local_data.csv", "train.csv", "data.csv")):
    """
    Loads dataset for a single bank/branch.
    Looks for {data_path}/{bank_id}/{one_of_filename_candidates}
    """
    folder = os.path.join(data_path, bank_id)

    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Bank folder not found: {folder}")

    # Find a file that exists
    path = None
    for fname in filename_candidates:
        candidate = os.path.join(folder, fname)
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"No dataset file found in {folder}. Tried: {filename_candidates}"
        )

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