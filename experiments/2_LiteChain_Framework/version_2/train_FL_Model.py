# train_FL_Model.py

import os
import time
import subprocess
import torch
import torch.nn as nn
import random
import math
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np  # For dataset simulation
from FL_model import LSTMTabular


# From dataset.py
def load_bank_dataset(num_samples=1000, input_dim=10):
    # Simulate bank fraud dataset: features like transaction amounts, etc.
    X = torch.randn(num_samples, 1, input_dim)  # Batch size 1 for simplicity
    y = torch.randint(0, 2, (num_samples,)).float()  # Binary labels (fraud/non-fraud) as float
    return list(zip(X, y))

# From clustering.utils
def get_client_id():
    # In Docker, set CLIENT_ID env var per container (e.g., bank1, bank2, etc.)
    return os.environ.get('CLIENT_ID', 'bank_1')  # Default to bank_1 if not set

# From clustering.clustering_game
class ClusteringGame:
    def __init__(self, client_ids: List[str], num_clusters: int = 2):
        self.client_ids = client_ids
        self.num_clusters = min(num_clusters, len(client_ids))
        # Simulate network metrics (latency, capability, reputation) for dynamic clustering
        self.client_metrics = {cid: {
            'latency': random.uniform(10, 100),  # ms
            'capability': random.uniform(0.5, 1.0),  # computational power
            'size': random.randint(100, 1000),  # data size
            'reputation': random.uniform(0.8, 1.0)
        } for cid in client_ids}
    
    def form_clusters(self) -> Dict[int, List[str]]:
        # Dynamic clustering: group by latency and capability (simplified from Algorithm1_DistributedNetworkOptimization)
        clusters = defaultdict(list)
        sorted_clients = sorted(self.client_metrics.items(), key=lambda x: x[1]['latency'])
        chunk_size = math.ceil(len(self.client_ids) / self.num_clusters)
        for i, (cid, _) in enumerate(sorted_clients):
            cluster_id = i // chunk_size
            clusters[cluster_id].append(cid)
        return dict(clusters)

# From clustering.cluster_state
class ClusterState:
    def __init__(self):
        self.clusters = {}  # cluster_id -> list of client_ids
        self.cluster_heads = {}  # cluster_id -> client_id
    
    def update_cluster(self, cluster_id: int, members: List[str]):
        self.clusters[cluster_id] = members
    
    def set_cluster_head(self, cluster_id: int, head_id: str):
        self.cluster_heads[cluster_id] = head_id
    
    def get_cluster_head(self, cluster_id: int) -> str:
        return self.cluster_heads.get(cluster_id)
    
    def get_cluster_members(self, cluster_id: int) -> List[str]:
        return self.clusters.get(cluster_id, [])

# From clustering.committee
class CommitteeElection:
    def __init__(self, metrics: Dict[str, Dict]):
        self.metrics = metrics
    
    def elect_cluster_head(self, members: List[str]) -> str:
        # Elect based on reputation and size (highest reputation, then size; from elect_committee_member_alg6)
        best = members[0]
        for cid in members[1:]:
            if (self.metrics[cid]['reputation'] > self.metrics[best]['reputation'] or
                (self.metrics[cid]['reputation'] == self.metrics[best]['reputation'] and
                 self.metrics[cid]['size'] > self.metrics[best]['size'])):
                best = cid
        return best

# From clustering.cluster_head
class ClusterHead:
    def __init__(self, cluster_id: int, head_id: str, metrics: Dict[str, Dict]):
        self.cluster_id = cluster_id
        self.head_id = head_id
        self.metrics = metrics
        self.received_updates = []
    
    def verify_update(self, update: Dict[str, torch.Tensor]) -> bool:
        # Basic verification: check if update has valid parameters (e.g., not empty, reasonable norms)
        if not update:
            return False
        total_norm = sum(torch.norm(p).item() for p in update.values())
        return 0 < total_norm < 1000  # Arbitrary threshold; in real setup, use validation accuracy
    
    def aggregate_updates(self, verified_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not verified_updates:
            return {}
        # FedAvg-style aggregation with staleness weighting (inspired by Algorithm 5)
        aggregated = {}
        total_weight = sum(self.metrics[cid]['size'] for cid in self.received_updates if self.verify_update(verified_updates[len(aggregated)]))
        for key in verified_updates[0].keys():
            aggregated[key] = sum(update[key] * self.metrics[cid]['size'] / total_weight
                                  for update, cid in zip(verified_updates, self.received_updates)
                                  if self.verify_update(update))
        return aggregated

def local_train(model, data, epochs=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    
    for _ in range(epochs):
        for x, y in data:
            x = x.unsqueeze(0)  # Reshape to (1, 1, input_dim) for LSTM
            y = y.unsqueeze(0)  # Reshape to (1,)
            optimizer.zero_grad()
            loss = loss_fn(model(x).squeeze(-1), y)
            loss.backward()
            optimizer.step()
    
    return model.state_dict()

def run_as_client():
    client_ids = ["bank_1", "bank_2", "bank_3", "bank_4", "bank_5", "bank_6", "bank_7", "bank_8", "bank_9", "bank_10"]
    current_client = get_client_id()
    
    # ---------------- CLUSTERING ----------------
    clustering = ClusteringGame(client_ids, num_clusters=2)  # Dynamic clustering
    clusters = clustering.form_clusters()
    
    cluster_state = ClusterState()
    committee = CommitteeElection(clustering.client_metrics)
    
    for cid, members in clusters.items():
        cluster_state.update_cluster(cid, members)
        head = committee.elect_cluster_head(members)
        cluster_state.set_cluster_head(cid, head)
        print(f"Cluster {cid}: Members {members}, Head {head}")
    
    # Find my cluster
    my_cluster_id = None
    for cid, members in cluster_state.clusters.items():
        if current_client in members:
            my_cluster_id = cid
            break
    
    if my_cluster_id is None:
        print(f"[{current_client}] Not in any cluster!")
        return
    
    cluster_head_id = cluster_state.get_cluster_head(my_cluster_id)
    
    # ---------------- LOCAL TRAINING ----------------
    model = LSTMTabular()
    data = load_bank_dataset()  # Load simulated dataset
    update = local_train(model, data)
    
    # ---------------- SEND UPDATE ----------------
    if current_client != cluster_head_id:
        # Send update only to cluster head (simulate IPC/networking in Docker)
        print(f"[{current_client}] Sending update to cluster head {cluster_head_id}")
        # In Docker: Use sockets, shared volumes, or message queues here
    
    else:
        # I'm the cluster head & committee member
        print(f"[{current_client}] Acting as cluster head for Cluster {my_cluster_id}")
        
        # Simulate receiving updates from other members (in real Docker, listen for messages)
        received_updates = [update]  # Add actual received updates here
        ch = ClusterHead(my_cluster_id, current_client, clustering.client_metrics)
        ch.received_updates = cluster_state.get_cluster_members(my_cluster_id)
        
        verified = [u for u in received_updates if ch.verify_update(u)]
        aggregated = ch.aggregate_updates(verified)
        
        print(f"[Cluster {my_cluster_id}] Verified {len(verified)}/{len(received_updates)} updates, Aggregated model ready")
        # In full system: Propagate aggregated model to inter-cluster or blockchain

def run_as_central():
    banks = [f"bank_{i}" for i in range(1, 11)]
    
    # Build the Docker image
    project_dir = "/media/fyp-group-18/1TB-Hard/FYP-Group18"
    subprocess.run(["docker", "build", "-f", "docker/Dockerfile.bank", "-t", "litechain_fl", "."], cwd=project_dir, check=True)
    
    print("Starting 10 Docker containers for banks...")
    for bank in banks:
        # Run container for each bank
        cmd = [
            "docker", "run", "--rm",
            "-e", f"CLIENT_ID={bank}",
            "-v", f"{project_dir}/data/processed/3_local_silo_balancing:/data",
            "-v", f"{project_dir}/logs:/logs",
            "litechain_fl"
        ]
        subprocess.run(cmd, cwd=project_dir)
    
    print("All containers finished. Clustering, training, and aggregation completed.")

def main():
    # If CLIENT_ID is set, run as client
    if os.environ.get('CLIENT_ID'):
        run_as_client()
        return
    
    # Else, run as central orchestrator
    run_as_central()
    
if __name__ == "__main__":
    main()