# src/aggregation/global_aggregate.py  (PHASE 2)
# Inter-brand aggregation (global node, Phase 2)

from __future__ import annotations
import os
import argparse
from typing import List

from src.clustering.topology_loader import load_topology
from src.clustering.ids import brand_model_filename, global_model_filename
from src.aggregation.fedavg import load_state_dict, save_state_dict, fedavg_state_dicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="config/topology.yaml")
    ap.add_argument("--round_dir", required=True, help="e.g., shared/round_0001")
    ap.add_argument("--brand_models_subdir", default="brand_models")
    args = ap.parse_args()

    topo = load_topology(args.topology)
    brand_models_dir = os.path.join(args.round_dir, args.brand_models_subdir)

    if not os.path.exists(brand_models_dir):
        raise FileNotFoundError(f"Brand models dir not found: {brand_models_dir}")

    state_dicts = []
    used_brands: List[str] = []

    for brand_id in topo.brand_to_branches.keys():
        path = os.path.join(brand_models_dir, brand_model_filename(brand_id))
        if os.path.exists(path):
            state_dicts.append(load_state_dict(path))
            used_brands.append(brand_id)

    if not state_dicts:
        raise RuntimeError("No brand models available (all brands excluded).")

    global_sd = fedavg_state_dicts(state_dicts)
    out_path = os.path.join(args.round_dir, global_model_filename())
    save_state_dict(global_sd, out_path)

    print(f"[GLOBAL-P2] used brands: {used_brands}")
    print(f"[GLOBAL-P2] wrote global model: {out_path}")


if __name__ == "__main__":
    main()