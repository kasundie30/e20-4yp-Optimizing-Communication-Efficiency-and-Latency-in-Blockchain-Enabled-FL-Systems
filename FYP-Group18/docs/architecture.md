# System Architecture

## Overview
The system architecture consists of four learning paradigms evaluated progressively:

1. Centralized Learning
2. Federated Learning (FL)
3. LiteChain-based BCFL
4. Proposed LiteChain-CLO Framework

## Components
- **Clients:** Perform local training on private transaction data.
- **Federated Server:** Aggregates model updates.
- **Blockchain Network:** Ensures transparency, auditability, and trust.
- **Consensus Mechanism:** Validates transactions and model updates.
- **Credit Scoring Module:** Evaluates client reliability.

## LiteChain-CLO Enhancements
- Latency-aware aggregation strategy
- Communication-efficient update propagation
- Adaptive client selection using credit scores

## Design Goals
- Reduce communication overhead
- Minimize end-to-end training latency
- Preserve model accuracy in highly imbalanced settings
