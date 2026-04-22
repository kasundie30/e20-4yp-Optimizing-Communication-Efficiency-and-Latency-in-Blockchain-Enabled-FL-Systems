import pandas as pd
import os
from imblearn.over_sampling import SMOTE
import glob

# Path where Dirichlet split saved the bank folders
silo_dir = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/2_bank_silos"
output_base = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/3_local_silo_balancing"

# Iterate through each bank folder (Bank_1, Bank_2, etc.)
bank_folders = [f for f in os.listdir(silo_dir) if os.path.isdir(os.path.join(silo_dir, f))]

for bank in bank_folders:
    print(f"--- Processing {bank} ---")
    
    # 1. Load the localized (Non-IID) data for this specific bank
    local_csv = os.path.join(silo_dir, bank, "local_data.csv")
    df = pd.read_csv(local_csv)
    
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # 2. Check if the bank actually has fraud cases (Dirichlet split might leave some with 0)
    if y.sum() < 6: # SMOTE typically needs at least 6 samples to find k-neighbors
        print(f"Skipping SMOTE for {bank}: Too few fraud samples for interpolation.")
        # Optional: Use RandomOverSampler here instead if fraud count is > 0 but < 6
        df_balanced = df
    else:
        # 3. Apply SMOTE LOCALLY using only this bank's perspective of fraud
        smote = SMOTE(random_state=42, k_neighbors=min(5, y.sum() - 1))
        X_res, y_res = smote.fit_resample(X, y)
        df_balanced = pd.concat([pd.DataFrame(X_res, columns=X.columns), 
                                pd.Series(y_res, name="Class")], axis=1)
        print(f"{bank} balanced: {y_res.value_counts().to_dict()}")

    # 4. Save to a training-ready folder
    bank_output_dir = os.path.join(output_base, bank)
    os.makedirs(bank_output_dir, exist_ok=True)
    df_balanced.to_csv(os.path.join(bank_output_dir, "train_ready.csv"), index=False)

print("\nSuccess! All banks have locally balanced datasets ready for LiteChain training.")