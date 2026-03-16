# src/run_rounds.py
# Orchestrates the entire multi-round federated learning process

from __future__ import annotations
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

from src.clustering.topology_loader import load_topology
from src.clustering.ids import global_model_filename


def run_cmd(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    print("\n[CMD]", " ".join(cmd))
    new_env = os.environ.copy()
    if env:
        new_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=new_env, check=True)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_global_model(src: Path, dst_round_dir: Path) -> Path:
    ensure_dir(dst_round_dir)
    dst = dst_round_dir / global_model_filename()
    shutil.copy2(src, dst)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", default="config/topology.yaml")
    ap.add_argument("--compose_file", default="docker/docker-compose.yml",
                    help="docker compose file that launches ALL branch services")
    ap.add_argument("--project_root", default=".",
                    help="repo root so docker compose paths resolve correctly")
    ap.add_argument("--shared_root", default="shared",
                    help="where round folders will be created")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--init_global_model", required=True,
                    help="path to initial global_model.pt for round_0001")

    args = ap.parse_args()

    topo = load_topology(args.topology)

    project_root = Path(args.project_root).resolve()
    shared_root = (project_root / args.shared_root).resolve()
    ensure_dir(shared_root)

    prev_global = Path(args.init_global_model).resolve()
    if not prev_global.exists():
        raise FileNotFoundError(f"Initial global model not found: {prev_global}")

    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] shared_root  = {shared_root}")
    print(f"[INFO] rounds       = {args.rounds}")
    print(f"[INFO] brands       = {list(topo.brand_to_branches.keys())}")

    for t in range(1, args.rounds + 1):
        round_dir = shared_root / f"round_{t:04d}"
        ensure_dir(round_dir)

        # (Optional) organize subfolders. local_train writes to /logs root,
        # so we keep it simple in Phase 1.
        ensure_dir(round_dir / "brand_models")

        # 1) Place the global model into this round folder (mounted to /logs)
        placed = copy_global_model(prev_global, round_dir)
        print(f"[ROUND {t:04d}] placed global model -> {placed}")

        # 2) Run all branch containers
        # Pass the current round folder (absolute path) to docker compose
        abs_round_dir = str(round_dir.resolve())
        run_cmd(["docker", "compose", "-f", str(project_root / args.compose_file),
                 "up", "--build", "--remove-orphans"],
                cwd=str(project_root),
                env={"LOGS_DIR": abs_round_dir})

        # 3) HQ aggregation (intra-cluster)
        run_cmd([sys.executable, "-m", "src.aggregation.hq_aggregate",
                 "--round_dir", str(round_dir),
                 "--topology", str(project_root / args.topology)],
                cwd=str(project_root))

        # 4) Global aggregation (inter-cluster)
        run_cmd([sys.executable, "-m", "src.aggregation.global_aggregate",
                 "--round_dir", str(round_dir),
                 "--topology", str(project_root / args.topology)],
                cwd=str(project_root))

        # 5) The new global model becomes prev_global for next round
        new_global = round_dir / global_model_filename()
        if not new_global.exists():
            raise FileNotFoundError(f"[ROUND {t:04d}] global model not produced: {new_global}")

        prev_global = new_global
        print(f"[ROUND {t:04d}] new global model -> {prev_global}")

        # Optional: stop containers if they remain running (depends on compose behavior)
        run_cmd(["docker", "compose", "-f", str(project_root / args.compose_file),
                 "down"], cwd=str(project_root))

    print("\n[DONE] All rounds completed.")


if __name__ == "__main__":
    main()