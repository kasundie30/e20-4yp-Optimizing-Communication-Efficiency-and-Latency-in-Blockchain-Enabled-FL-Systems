# Dataset and Preprocessing

## Dataset Summary
The European Credit Card Fraud Dataset is highly imbalanced, making it suitable for evaluating adaptive and robust fraud detection models.

## Preprocessing Steps
- Feature normalization (Amount, Time)
- Handling class imbalance using:
  - Undersampling
  - Oversampling / SMOTE (where applicable)
- Train-test splitting

## Federated Data Distribution
- Client datasets are generated from the processed dataset.
- Both IID and non-IID distributions are supported.
- Each client simulates an independent financial entity.

## Evaluation Considerations
Due to class imbalance, performance is measured using:
- Precision
- Recall
- F1-score
- ROC-AUC
