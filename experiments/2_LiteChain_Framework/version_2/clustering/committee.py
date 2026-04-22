# committee.py

import random

class CommitteeElection:
    def __init__(self):
        pass

    def elect_cluster_head(self, cluster_members):
        """
        Cluster head is also committee member
        Dynamic per round
        """
        return random.choice(cluster_members)
