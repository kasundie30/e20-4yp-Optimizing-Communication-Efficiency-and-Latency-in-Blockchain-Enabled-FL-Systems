"""
fl-layer/model/dataset.py
Cleaned dataset loader ported from CCFD-FL-layer/dataset.py.

Changes vs. source:
  - data_path is a required parameter (no hidden CD-layer default)
  - partition_index / num_partitions allow non-overlapping splits
  - raises FileNotFoundError clearly (original did this, preserved)
  - no hardcoded paths, no CCFD-FL-layer imports
"""
import logging
import os
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

FILENAME_CANDIDATES: Tuple[str, ...] = (
    "train_ready.csv",
    "local_data.csv",
    "train.csv",
    "data.csv",
)


def load_bank_dataset(
    bank_id: str,
    data_path: str,
    filename_candidates: Sequence[str] = FILENAME_CANDIDATES,
    partition_index: Optional[int] = None,
    num_partitions: Optional[int] = None,
) -> TensorDataset:
    """
    Load and optionally partition a bank/branch dataset.

    Args:
        bank_id            : sub-folder name inside data_path (e.g. 'BankA')
        data_path          : root directory that contains bank sub-folders
        filename_candidates: ordered list of CSV filenames to try
        partition_index    : 0-based partition to return (None = whole dataset)
        num_partitions     : total number of partitions (required if partition_index set)

    Returns:
        TensorDataset of (X_float32, y_float32_col)

    Raises:
        FileNotFoundError  : if the bank folder or no CSV is found
        ValueError         : if partition args are inconsistent
    """
    folder = os.path.join(data_path, bank_id)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Bank folder not found: {folder}")

    path: Optional[str] = None
    for fname in filename_candidates:
        candidate = os.path.join(folder, fname)
        if os.path.exists(candidate):
            path = candidate
            break

    if path is None:
        raise FileNotFoundError(
            f"No dataset file found in {folder}. Tried: {list(filename_candidates)}"
        )

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Apply partitioning before scaling to avoid data leakage across partitions
    if partition_index is not None:
        if num_partitions is None or num_partitions <= 0:
            raise ValueError("num_partitions must be a positive int when partition_index is given")
        if not (0 <= partition_index < num_partitions):
            raise ValueError(
                f"partition_index {partition_index} out of range for num_partitions={num_partitions}"
            )
        n = len(X)
        size = n // num_partitions
        start = partition_index * size
        end = start + size if partition_index < num_partitions - 1 else n
        X = X[start:end]
        y = y[start:end]
        logger.debug("Partition %d/%d: rows %d–%d", partition_index, num_partitions, start, end)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    logger.debug("Dataset loaded: %d samples, %d features", len(X_tensor), X_tensor.shape[1])
    return TensorDataset(X_tensor, y_tensor)
