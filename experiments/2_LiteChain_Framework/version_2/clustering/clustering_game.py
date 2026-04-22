# clustering_game.py

import random

class ClusteringGame:
    def __init__(self, client_ids, max_cluster_size=3):
        self.client_ids = client_ids
        self.max_cluster_size = max_cluster_size

    def form_clusters(self):
        """
        Devices choose 'switch' or 'remain' strategy
        Simplified bi-objective optimization:
        - minimize cluster size imbalance
        - maximize locality (randomized here, replace with real metrics later)
        """
        shuffled = self.client_ids[:]
        random.shuffle(shuffled)

        clusters = {}
        cluster_id = 0

        for client in shuffled:
            placed = False
            for cid, members in clusters.items():
                if len(members) < self.max_cluster_size:
                    members.append(client)
                    placed = True
                    break
            if not placed:
                clusters[cluster_id] = [client]
                cluster_id += 1

        return clusters
