import pandas as pd
import os
from imblearn.over_sampling import SMOTE

# Load the scaled data from the previous step
input_path = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/1_feature_scaled/creditcard_scaled_time_dropped.csv"
output_dir = "/scratch1/e20-fyp-blockchain-fraud-detect/e20-4yp-Optimizing-Communication-Efficiency-and-Latency-in-Blockchain-Enabled-FL-Systems/FYP-Group18/data/processed/2_balanced/"
os.makedirs(output_dir, exist_ok=True)

print("Loading scaled data...")
df = pd.read_csv(input_path)
X = df.drop("Class", axis=1)
y = df["Class"]

# applying SMOTE to the entire dataset before it is split into banks!!
print("Applying SMOTE (this may take a moment)...")
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

df_balanced = pd.concat([pd.DataFrame(X_res), pd.Series(y_res, name="Class")], axis=1)
df_balanced.to_csv(os.path.join(output_dir, "creditcard_balanced_time_dropped.csv"), index=False)
print(f"Success! Balanced distribution:\n{y_res.value_counts()}")