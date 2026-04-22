# cluster_state.py

class ClusterState:
    def __init__(self):
        self.clusters = {}        # cluster_id -> list of client_ids
        self.cluster_heads = {}   # cluster_id -> client_id

    def update_cluster(self, cluster_id, members):
        self.clusters[cluster_id] = members

    def set_cluster_head(self, cluster_id, head_id):
        self.cluster_heads[cluster_id] = head_id

    def get_cluster_head(self, cluster_id):
        return self.cluster_heads.get(cluster_id)

    def get_cluster_members(self, cluster_id):
        return self.clusters.get(cluster_id, [])
