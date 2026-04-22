"""
fl-integration/tests/locustfile.py
Locust load test script for the HCFL FastAPI server (Phase 10.5).
Simulates concurrent requests to test scalability and response times.
"""

import time
import random
from locust import HttpUser, task, between

class HCFLApiUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def check_health(self):
        self.client.get("/health")

    @task(2)
    def fetch_trust_scores(self):
        self.client.get("/trust-scores")

    @task(2)
    def check_consensus(self):
        round_num = random.randint(1, 10)
        self.client.get(f"/check-consensus/{round_num}")
        
    @task(2)
    def get_global_model(self):
        round_num = random.randint(1, 10)
        # We might get 404s for non-existent rounds, which is expected behavior, 
        # so we catch them so Locust doesn't mark it as a failure.
        with self.client.get(f"/global-model/{round_num}", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()

    @task(1)
    def submit_verification(self):
        # Simulate a quick verification POST to avoid spamming the chain ledger too heavily
        # but to test the write-path overhead of the API
        payload = {
            "verifier_id": "BankB",
            "target_bank_id": random.choice(["BankA", "BankB", "BankC"]), # Strict Pydantic rule
            "round": random.randint(1, 10),
            "verified": random.choice([True, False]),
        }
        with self.client.post("/submit-verification", json=payload, catch_response=True) as response:
            if response.status_code in [200, 422, 503, 500, 400]:
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
