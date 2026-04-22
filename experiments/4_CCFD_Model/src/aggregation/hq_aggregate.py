# src/aggregation/hq_aggregate.py  (PHASE 2)
# Intra-brand aggregation (HQ node, Phase 2)

from __future__ import annotations
import os
import argparse
from typing import List

from src.clustering.topology_loader import load_topology
from src.clustering.ids import brand_model_filename
from src.aggregation.fedavg import save_state_dict, fedavg_state_dicts

from src.utils.config_loader import load_config
from src.resilience.deadline_collect import collect_until_deadline
from src.validation.validate_fast import fast_validate_state_dict
from src.resilience.backup_logic import blend_with_prev_global
from src.utils.score_store import load_scores, save_scores, update_score
from src.utils.ledger import append_record, hash_state_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="config/topology.yaml")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--round_dir", required=True, help="e.g., shared/round_0001")
    ap.add_argument("--out_subdir", default="brand_models")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p2 = cfg.get("phase2", {})
    scoring_cfg = cfg.get("scoring", {})
    paths_cfg = cfg.get("paths", {})

    deadline_sec = int(p2.get("deadline_sec", 25))
    min_required = int(p2.get("min_models_required", 2))
    data_root = str(p2.get("data_root", "data/processed/3_local_silo_balancing"))
    frac = float(p2.get("fast_val_fraction", 0.15))
    metric_name = str(p2.get("metric", "prauc"))
    thresholds = p2.get("threshold", {})
    thr = float(thresholds.get(metric_name, 0.2))
    beta = float(p2.get("blend_beta", 0.7))

    score_file = paths_cfg.get("scores_file", "shared/scores.json")
    ledger_file = paths_cfg.get("ledger_file", "shared/ledger.jsonl")

    init_score = float(scoring_cfg.get("init_score", 1.0))
    floor = float(scoring_cfg.get("floor", 0.2))
    سق_max = float(scoring_cfg.get("max", 3.0))
    reward = float(scoring_cfg.get("reward", 0.05))
    penalty = float(scoring_cfg.get("penalty", 0.10))

    topo = load_topology(args.topology)
    out_dir = os.path.join(args.round_dir, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    brands = list(topo.brand_to_branches.keys())
    scores = load_scores(score_file, brands, init_score)

    print(f"[HQ-P2] round_dir={args.round_dir}")
    print(f"[HQ-P2] deadline_sec={deadline_sec} min_required={min_required}")
    print(f"[HQ-P2] validation metric={metric_name} thr={thr} frac={frac}")
    print(f"[HQ-P2] data_root={data_root}")
    print(f"[HQ-P2] output={out_dir}")

    for brand_id, branch_ids in topo.brand_to_branches.items():
        hq_id = topo.brand_to_hq[brand_id]
        arrived_ids, state_dicts = collect_until_deadline(args.round_dir, branch_ids, deadline_sec)

        if len(state_dicts) == 0:
            print(f"[HQ-P2] brand={brand_id}: no branch models arrived -> EXCLUDED")
            update_score(scores, brand_id, False, reward, penalty, floor, سق_max)
            append_record(ledger_file, {
                "brand": brand_id,
                "status": "excluded_no_updates",
                "arrived_branches": arrived_ids,
                "score": scores[brand_id],
            })
            continue

        # if fewer than min_required, still aggregate but mark as risky (backup-like behavior)
        ok_collection = (len(state_dicts) >= min_required)

        brand_sd = fedavg_state_dicts(state_dicts)

        # fast validate
        metrics = fast_validate_state_dict(
            state_dict=brand_sd,
            node_id_for_data=hq_id,
            data_root=data_root,
            fraction=frac,
            device="cpu"
        )
        passed = metrics.get(metric_name, 0.0) >= thr

        # backup recovery if failed validation OR weak collection
        if (not passed) or (not ok_collection):
            brand_sd_alt = blend_with_prev_global(args.round_dir, brand_sd, beta=beta)
            metrics_alt = fast_validate_state_dict(
                state_dict=brand_sd_alt,
                node_id_for_data=hq_id,
                data_root=data_root,
                fraction=frac,
                device="cpu"
            )
            passed_alt = metrics_alt.get(metric_name, 0.0) >= thr

            if passed_alt:
                brand_sd = brand_sd_alt
                metrics = metrics_alt
                passed = True

        if not passed:
            print(f"[HQ-P2] brand={brand_id}: validation failed -> EXCLUDED | metrics={metrics}")
            update_score(scores, brand_id, False, reward, penalty, floor, سق_max)
            append_record(ledger_file, {
                "brand": brand_id,
                "status": "excluded_validation_fail",
                "arrived_branches": arrived_ids,
                "metrics": metrics,
                "score": scores[brand_id],
            })
            continue

        # accepted: save brand model
        out_path = os.path.join(out_dir, brand_model_filename(brand_id))
        save_state_dict(brand_sd, out_path)

        update_score(scores, brand_id, True, reward, penalty, floor, سق_max)

        append_record(ledger_file, {
            "brand": brand_id,
            "status": "accepted",
            "arrived_branches": arrived_ids,
            "metrics": metrics,
            "score": scores[brand_id],
            "brand_hash": hash_state_dict(brand_sd),
            "brand_model_path": out_path,
        })

        print(f"[HQ-P2] brand={brand_id}: ACCEPTED -> {out_path} | metrics={metrics} | score={scores[brand_id]:.3f}")

    save_scores(score_file, scores)
    print("[HQ-P2] done.")


if __name_
_ == "__main__":
    main()