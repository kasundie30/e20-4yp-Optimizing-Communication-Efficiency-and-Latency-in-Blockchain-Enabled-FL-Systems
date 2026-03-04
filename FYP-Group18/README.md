# LiteChain-CLO  
**A Communication and Latency-Optimized LiteChain Framework for Adaptive Fraud Detection**

## Overview
LiteChain-CLO is an enhanced blockchain-based federated learning (BCFL) framework designed to optimize communication overhead and reduce training latency for adaptive credit card fraud detection. The framework builds upon the LiteChain architecture by incorporating latency-aware aggregation, adaptive credit mechanisms, and communication-efficient strategies.

## Research Pipeline
The experimental workflow follows four progressive stages:

1. Centralized Fraud Detection Model  
2. Federated Learning (FL) Baseline  
3. LiteChain-based BCFL Baseline  
4. Proposed LiteChain-CLO Framework  

## Dataset
- **European Credit Card Fraud Dataset**
- 284,807 transactions
- 0.17% fraudulent (highly imbalanced)
- Widely used benchmark in fraud detection research

## Key Contributions
- Communication-aware federated learning over blockchain
- Latency-optimized consensus and aggregation mechanisms
- Adaptive client credit scoring for reliable fraud detection
- Extensive evaluation against centralized, FL, and BCFL baselines

## Project Structure
Each folder is organized to support reproducibility, extensibility, and research clarity.

## Reproducibility
All experiments are configurable via YAML files under `experiments/`.  
Logs, model checkpoints, and result artifacts are stored separately.

## License
This project is intended for academic research purposes.
