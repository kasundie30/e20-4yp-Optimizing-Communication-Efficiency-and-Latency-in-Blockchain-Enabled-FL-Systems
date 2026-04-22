"""
Primary Metrics for Federated Learning Evaluation

This module provides functions to calculate primary metrics such as:
- Communication Overhead
- End-to-End Latency
- Convergence Speed
"""

import time
import numpy as np

def calculate_communication_overhead(model_sizes, num_clients, num_rounds):
    """
    Calculate the total communication overhead in bytes.

    Parameters:
    - model_sizes (list): List of model sizes in bytes for each round or per client.
    - num_clients (int): Number of clients participating.
    - num_rounds (int): Number of federated learning rounds.

    Returns:
    - float: Total communication overhead.
    """
    # Assuming each client sends the model size each round
    if isinstance(model_sizes, list):
        # If model_sizes is a list, assume it's per round or average
        avg_size = np.mean(model_sizes)
    else:
        avg_size = model_sizes
    return avg_size * num_clients * num_rounds

def calculate_end_to_end_latency(start_time, end_time):
    """
    Calculate the end-to-end latency in seconds.

    Parameters:
    - start_time (float): Start time (from time.time()).
    - end_time (float): End time (from time.time()).

    Returns:
    - float: Latency in seconds.
    """
    return end_time - start_time

def calculate_convergence_speed(losses, threshold=1e-4):
    """
    Calculate the number of rounds to converge based on loss decrease.

    Parameters:
    - losses (list): List of loss values per round.
    - threshold (float): Threshold for convergence (e.g., change in loss).

    Returns:
    - int: Number of rounds to converge.
    """
    for i in range(1, len(losses)):
        if abs(losses[i] - losses[i-1]) < threshold:
            return i
    return len(losses)  # If not converged, return total rounds