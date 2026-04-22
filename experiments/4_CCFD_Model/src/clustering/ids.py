# src/clustering/ids.py
# Helper functions for naming and file paths

def make_branch_id(brand_id: str, branch_idx: int) -> str:
    return f"{brand_id}_branch_{branch_idx}"

def local_model_filename(branch_id: str) -> str:
    # local_train.py saves: /logs/{BANK_ID}_local_model.pt
    return f"{branch_id}_local_model.pt"

def brand_model_filename(brand_id: str) -> str:
    return f"{brand_id}.pt"

def global_model_filename() -> str:
    return "global_model.pt"