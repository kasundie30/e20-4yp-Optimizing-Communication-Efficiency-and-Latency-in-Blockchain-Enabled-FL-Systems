"""All 8 evaluation metrics: PR-AUC, ROC-AUC, F1, Precision, Recall, Comm MB, E2E sec."""
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_fscore_support)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.lstm_model import LSTMTabular
import config

def predict_proba(state_dict, X, batch_size=512):
    model = LSTMTabular(config.INPUT_DIM, config.HIDDEN_DIM, config.NUM_LAYERS)
    model.load_state_dict(state_dict)
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.unsqueeze(1)).squeeze(1)
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out)

def classification_metrics(y_true, y_prob, threshold=config.THRESHOLD):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prauc  = float(average_precision_score(y_true, y_prob))
        rocauc = float(roc_auc_score(y_true, y_prob))
        y_pred = (y_prob >= threshold).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred,
                                                       average="binary", zero_division=0)
    return {"prauc": prauc, "rocauc": rocauc, "f1": float(f1),
            "precision": float(p), "recall": float(r)}

def communication_overhead_mb(upload_bytes_list, global_bytes, num_clients):
    """Upload = sum of all local model sizes; Download = global × num_clients."""
    return (sum(upload_bytes_list) + global_bytes * num_clients) / (1024**2)

def end_to_end_latency_sec(client_times, agg_time):
    """E2E = max(parallel client train times) + aggregation time."""
    return max(client_times) + agg_time
