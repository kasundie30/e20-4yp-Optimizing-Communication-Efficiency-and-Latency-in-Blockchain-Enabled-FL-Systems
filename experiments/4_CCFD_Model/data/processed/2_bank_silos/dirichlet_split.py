import numpy as np
import pandas as pd
import os

def dirichlet_split_noniid(df, alpha, n_clients):
    """Splits data into non-IID partitions using Dirichlet distribution."""
    n_classes = df['Class'].nunique()
    # Create an empty list for each bank's indices
    client_indices = [[] for _ in range(n_clients)]
    
    for k in range(n_classes):
        label_indices = np.where(df['Class'] == k)[0]
        np.random.shuffle(label_indices)
        
        # Dirichlet distribution gives us the proportions for each bank
        proportions = np.random.dirichlet([alpha] * n_clients)
        
        # Split the indices based on proportions
        proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        split_indices = np.split(label_indices, proportions)
        
        for i in range(n_clients):
            client_indices[i].extend(split_indices[i])

    return client_indices

# --- IMPLEMENTATION ---
input_file = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/1_feature_scaled/creditcard_scaled_time_dropped.csv"
output_base = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/2_bank_silos/"
df = pd.read_csv(input_file)

# alpha=0.2 makes it very non-IID (Realistic)
indices = dirichlet_split_noniid(df, alpha=0.4, n_clients=10) 

for i, idx in enumerate(indices):
    bank_df = df.iloc[idx]
    bank_dir = os.path.join(output_base, f"bank_{i+1}")
    os.makedirs(bank_dir, exist_ok=True)
    bank_df.to_csv(os.path.join(bank_dir, "local_data.csv"), index=False)
    print(f"Bank {i+1} saved with {len(bank_df)} records.")