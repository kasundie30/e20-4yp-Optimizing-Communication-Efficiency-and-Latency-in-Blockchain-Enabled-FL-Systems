# cluster_head.py

import torch

class ClusterHead:
    def __init__(self, cluster_id, head_id):
        self.cluster_id = cluster_id
        self.head_id = head_id

    def verify_update(self, model_update):
        """
        Placeholder for:
        - signature check
        - norm bounds
        - anomaly detection
        """
        return True

    def aggregate_updates(self, updates):
        """
        Federated averaging inside cluster
        """
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = torch.mean(
                torch.stack([u[key] for u in updates]), dim=0
            )
        return aggregated
