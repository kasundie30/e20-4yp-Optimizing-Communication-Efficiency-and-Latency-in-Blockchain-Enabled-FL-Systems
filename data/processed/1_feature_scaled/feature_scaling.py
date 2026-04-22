import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# Update this path to where your raw CSV actually is
input_path = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/raw/creditcard.csv"
output_dir = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/1_feature_scaled/"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
df = pd.read_csv(input_path).drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))

# 1. DROP TIME: It is non-informative for a general FL fraud model
# 2. SEPARATE TARGET: Keep 'Class' aside
X = df.drop(columns=["Class", "Time"]) 
y = df["Class"]

print(f"Scaling {len(X.columns)} features (V1-V28 and Amount)...")

# 3. SCALE: This standardizes Amount and ensures V-features stay normalized
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 4. RECOMBINE: Combine scaled features with the original Class labels
df_scaled = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)

# Save the result
df_scaled.to_csv(os.path.join(output_dir, "creditcard_scaled_time_dropped.csv"), index=False)
print(f"Success! Scaled data (minus Time) saved to {output_dir}")