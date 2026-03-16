# src/clustering/topology_loader.py
# Loads static clusters (brands) and branches

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import yaml
import os


@dataclass
class Topology:
    brands: Dict[str, dict]                 # raw brand config
    branch_to_brand: Dict[str, str]         # branch_id -> brand_id
    brand_to_branches: Dict[str, List[str]] # brand_id -> [branch_ids]
    brand_to_hq: Dict[str, str]             # brand_id -> hq_branch_id
    brand_to_backup: Dict[str, str]         # brand_id -> backup_branch_id


def load_topology(path: str = "config/topology.yaml") -> Topology:
    """
    Loads static clusters (brands) and branches.
    Validates structure and returns convenient lookup maps.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Topology file not found: {path}")

    with open(path, "r") as f:
        topo = yaml.safe_load(f)

    if not topo or "brands" not in topo:
        raise ValueError("Invalid topology.yaml: missing top-level key 'brands'")

    brands = topo["brands"]
    if not isinstance(brands, dict) or len(brands) == 0:
        raise ValueError("Invalid topology.yaml: 'brands' must be a non-empty mapping")

    branch_to_brand: Dict[str, str] = {}
    brand_to_branches: Dict[str, List[str]] = {}
    brand_to_hq: Dict[str, str] = {}
    brand_to_backup: Dict[str, str] = {}

    all_branches_seen = set()

    for brand_id, info in brands.items():
        if "hq" not in info or "backup" not in info or "branches" not in info:
            raise ValueError(
                f"Brand '{brand_id}' must contain keys: hq, backup, branches"
            )

        branches = info["branches"]
        if not isinstance(branches, list) or len(branches) < 1:
            raise ValueError(f"Brand '{brand_id}' has empty/invalid branches list")

        hq = info["hq"]
        backup = info["backup"]

        if hq not in branches:
            raise ValueError(f"Brand '{brand_id}': hq '{hq}' must be included in branches")
        if backup not in branches:
            raise ValueError(f"Brand '{brand_id}': backup '{backup}' must be included in branches")

        # ensure no duplicate branch IDs across brands
        for br in branches:
            if br in all_branches_seen:
                raise ValueError(f"Duplicate branch id found across brands: '{br}'")
            all_branches_seen.add(br)
            branch_to_brand[br] = brand_id

        brand_to_branches[brand_id] = branches
        brand_to_hq[brand_id] = hq
        brand_to_backup[brand_id] = backup

    return Topology(
        brands=brands,
        branch_to_brand=branch_to_brand,
        brand_to_branches=brand_to_branches,
        brand_to_hq=brand_to_hq,
        brand_to_backup=brand_to_backup,
    )