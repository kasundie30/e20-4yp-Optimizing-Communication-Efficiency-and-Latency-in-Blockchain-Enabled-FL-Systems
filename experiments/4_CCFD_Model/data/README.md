# Dataset Overview

## Dataset Description
This project uses the **European Credit Card Fraud Dataset**, which contains real-world credit card transactions made by European cardholders.

- Total transactions: 284,807
- Fraudulent transactions: 492 (0.17%)
- Features: 30 (V1–V28 PCA-transformed, Time, Amount)
- Label: Class (0 = normal, 1 = fraud)

## Folder Structure
- `raw/`  
  Original dataset without any modification.

- `processed/`  
  Cleaned, normalized, balanced, or feature-engineered versions of the dataset.

- `splits/`  
  Dataset partitions for centralized, federated, and blockchain-based experiments.

## Data Handling
- No modifications are made to the raw dataset.
- Preprocessing scripts generate processed datasets reproducibly.
- Client-level splits support IID and non-IID federated learning scenarios.

## Ethical Considerations
The dataset is anonymized and publicly available, ensuring no personal or sensitive information is exposed.
