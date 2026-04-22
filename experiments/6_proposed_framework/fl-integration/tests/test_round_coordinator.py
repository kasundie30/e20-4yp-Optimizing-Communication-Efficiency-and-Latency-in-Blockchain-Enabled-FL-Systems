"""fl-integration/tests/test_round_coordinator.py — unit tests for RoundCoordinator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "fl-layer"))

import time
from typing import List, Tuple, Dict
from unittest.mock import MagicMock
import torch
import pytest

from model.FL_model import LSTMTabular
from round_coordinator import RoundCoordinator, RoundConfig, RoundResult


StateDict = Dict[str, torch.Tensor]


def _make_updates(n_branches=3) -> List[Tuple[StateDict, int]]:
    m = LSTMTabular(input_dim=30)
    return [(m.state_dict(), 100)] * n_branches


# -------------------------------------------------------
# Tests: deadline collection (6.4)
# -------------------------------------------------------

def test_deadline_returns_partial_when_slow():
    """Only 2/3 branches available → coordinator collects 2 before deadline."""
    available = _make_updates(n_branches=2)

    config = RoundConfig(
        round_num=1,
        bank_ids=["BankA", "BankB", "BankC"],
        deadline_sec=0.5,
        poll_interval=0.1,
    )
    coord = RoundCoordinator(config=config, collect_fn=lambda: available)
    updates = coord.collect_branch_updates()
    assert len(updates) == 2


def test_all_collected_before_deadline():
    """All 3 branches available immediately → returns early."""
    updates = _make_updates(n_branches=3)
    config = RoundConfig(
        round_num=1,
        bank_ids=["BankA", "BankB", "BankC"],
        deadline_sec=10.0,
        poll_interval=0.05,
    )
    start = time.time()
    coord = RoundCoordinator(config=config, collect_fn=lambda: updates)
    result = coord.collect_branch_updates()
    elapsed = time.time() - start
    assert len(result) == 3
    assert elapsed < 2.0, f"Should return early; took {elapsed:.2f}s"


# -------------------------------------------------------
# Tests: full run() with mocked HQ agents (6.5 integration)
# -------------------------------------------------------

def _make_hq_agent_mock(bank_id: str, submitted=True):
    agent = MagicMock()
    agent.run_round.return_value = {
        "val_score":  0.82,
        "model_cid":  f"Qm{bank_id}",
        "model_hash": "a" * 64,
        "submitted":  submitted,
        "tx":         {"tx_id": f"tx-{bank_id}"},
    }
    return agent


def test_run_returns_round_result():
    updates = _make_updates(3)
    config = RoundConfig(
        round_num=2,
        bank_ids=["BankA", "BankB", "BankC"],
        deadline_sec=5.0,
        poll_interval=0.05,
    )
    agents = {
        "BankA": _make_hq_agent_mock("BankA"),
        "BankB": _make_hq_agent_mock("BankB"),
        "BankC": _make_hq_agent_mock("BankC"),
    }
    coord = RoundCoordinator(config=config, collect_fn=lambda: updates, hq_agents=agents)
    result = coord.run()

    assert isinstance(result, RoundResult)
    assert result.round_num == 2
    assert result.collected == 3
    assert set(result.submitted_banks) == {"BankA", "BankB", "BankC"}
    assert result.elapsed_sec > 0


def test_run_no_updates_aborts():
    config = RoundConfig(round_num=1, bank_ids=["BankA"], deadline_sec=0.3, poll_interval=0.05)
    coord = RoundCoordinator(config=config, collect_fn=lambda: [])
    result = coord.run()
    assert result.collected == 0
    assert result.submitted_banks == []


def test_run_only_submitted_agents_counted():
    updates = _make_updates(2)
    config = RoundConfig(round_num=3, bank_ids=["BankA", "BankB"], deadline_sec=5.0, poll_interval=0.05)
    agents = {
        "BankA": _make_hq_agent_mock("BankA", submitted=True),
        "BankB": _make_hq_agent_mock("BankB", submitted=False),  # BankB fails
    }
    coord = RoundCoordinator(config=config, collect_fn=lambda: updates, hq_agents=agents)
    result = coord.run()
    assert "BankA" in result.submitted_banks
    assert "BankB" not in result.submitted_banks


def test_run_with_global_aggregator():
    """Global aggregation is only invoked when is_aggregator=True and submissions exist."""
    updates = _make_updates(2)
    config = RoundConfig(
        round_num=4,
        bank_ids=["BankA", "BankB"],
        deadline_sec=5.0,
        poll_interval=0.05,
        is_aggregator=True,
    )
    agents = {
        "BankA": _make_hq_agent_mock("BankA"),
        "BankB": _make_hq_agent_mock("BankB"),
    }
    mock_agg = MagicMock()
    mock_agg.run_full_aggregation.return_value = {
        "global_cid":  "QmGlobalFinal",
        "global_hash": "g" * 64,
        "weights":     {"BankA": 1.0, "BankB": 0.8},
    }
    coord = RoundCoordinator(
        config=config, collect_fn=lambda: updates,
        hq_agents=agents, global_agg=mock_agg,
    )
    result = coord.run()
    mock_agg.run_full_aggregation.assert_called_once_with(4, coord._bank_updates)
    assert result.global_cid == "QmGlobalFinal"
