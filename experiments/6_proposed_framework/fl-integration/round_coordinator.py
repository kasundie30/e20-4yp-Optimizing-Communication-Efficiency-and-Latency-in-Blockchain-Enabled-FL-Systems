"""
fl-integration/round_coordinator.py
Round Coordinator — manages timing and synchronization for one FL round. (6.4)

Responsibilities:
  - Set a deadline for branch updates to arrive (configurable)
  - Collect branch state_dicts (via injectable collect_fn matching Phase 5 pattern)
  - Dispatch each bank's run_round() call
  - Drive global aggregation if this node is the aggregator

Design: All timing is injectable so unit tests run in milliseconds.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fl-layer"))

from resilience.deadline_collect import wait_for_submissions
from config.config_loader import load_config

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]
CollectFn = Callable[[], List[Tuple[StateDict, int]]]   # returns [(state_dict, n_samples)]


@dataclass
class RoundConfig:
    """Configuration for a single FL round."""
    round_num:          int
    bank_ids:           List[str]
    deadline_sec:       float | None = None
    poll_interval:      float        = 1.0
    is_aggregator:      bool        = False     # True for the aggregator bank (BankA)
    aggregator_bank_id: str         = "BankA"


@dataclass
class RoundResult:
    """Outcome of a single FL round."""
    round_num:       int
    collected:       int            # number of branch updates collected
    submitted_banks: List[str]      = field(default_factory=list)
    global_cid:      Optional[str]  = None
    global_hash:     Optional[str]  = None
    elapsed_sec:     float          = 0.0
    timed_out:       bool           = False


class RoundCoordinator:
    """
    Coordinates timing, collection, and handoff for a single FL round.

    Args:
        config         : RoundConfig dataclass
        collect_fn     : callable() -> list[(state_dict, n_samples)]
                         Phase 5 deadline_collect or a live polling function
        hq_agents      : {bank_id: HQAgent} — one per participating bank
        global_agg     : GlobalAggregator instance (only used if config.is_aggregator)
        bank_updates   : {bank_id: {"model_cid", "model_hash", "num_samples"}}
                         pre-filled by HQ agents before aggregation
    """

    def __init__(
        self,
        config: RoundConfig,
        collect_fn: CollectFn,
        hq_agents: dict | None = None,
        global_agg: Any = None,
    ):
        self.config = config
        self.collect_fn = collect_fn
        self.hq_agents  = hq_agents or {}
        self.global_agg = global_agg
        self._bank_updates: Dict[str, dict] = {}
        
        # Load unified config and set deadline if not bounded
        fl_conf = load_config()
        if self.config.deadline_sec is None:
            self.config.deadline_sec = fl_conf.fl.deadline_seconds

    # ------------------------------------------------------------------
    # 6.4 — Round synchronization
    # ------------------------------------------------------------------

    def collect_branch_updates(self) -> List[Tuple[StateDict, int]]:
        """
        Wait up to deadline_sec for branch updates (state_dicts).
        Uses the Phase 5 deadline_collect.wait_for_submissions internally.

        Returns:
            List of (state_dict, num_samples) pairs collected before deadline.
        """
        expected = len(self.config.bank_ids)
        logger.info(
            "Waiting up to %.0fs for %d branch update(s) …",
            self.config.deadline_sec, expected,
            extra={"round_num": self.config.round_num, "component": "round_coordinator"}
        )
        updates = wait_for_submissions(
            expected_count=expected,
            collect_fn=self.collect_fn,
            deadline_sec=self.config.deadline_sec,
            poll_interval=self.config.poll_interval,
        )
        collected = len(updates)
        timed_out = collected < expected
        if timed_out:
            logger.warning(
                "Deadline reached — collected %d/%d updates.",
                collected, expected,
                extra={"round_num": self.config.round_num, "component": "round_coordinator"}
            )
        else:
            logger.info("All %d update(s) collected.", collected, extra={"round_num": self.config.round_num, "component": "round_coordinator"})
        return updates

    def run(self) -> RoundResult:
        """
        Execute a full FL round:
          1. Collect branch updates (with deadline)
          2. For each configured HQ agent, call run_round()
          3. If this node is the aggregator, run global aggregation

        Returns:
            RoundResult dataclass
        """
        start = time.time()
        result = RoundResult(round_num=self.config.round_num, collected=0)

        # Step 1: collect branch updates
        branch_updates = self.collect_branch_updates()
        result.collected = len(branch_updates)
        result.timed_out = result.collected < len(self.config.bank_ids)

        if result.collected == 0:
            logger.error("No updates collected — aborting.", extra={"round_num": self.config.round_num, "component": "round_coordinator"})
            result.elapsed_sec = time.time() - start
            return result

        # Step 2: HQ agents run FedAvg + submit to blockchain
        for bank_id, agent in self.hq_agents.items():
            try:
                out = agent.run_round(self.config.round_num, branch_updates)
                if out.get("submitted"):
                    result.submitted_banks.append(bank_id)
                    # Record for aggregator
                    self._bank_updates[bank_id] = {
                        "model_cid":   out["model_cid"],
                        "model_hash":  out["model_hash"],
                        "num_samples": result.collected,
                    }
            except Exception as e:
                logger.error("HQ agent %s failed: %s", bank_id, e, extra={"round_num": self.config.round_num, "component": "round_coordinator"})

        # Step 3: Global aggregation (only on aggregator node)
        if self.config.is_aggregator and self.global_agg and self._bank_updates:
            try:
                agg_result = self.global_agg.run_full_aggregation(
                    self.config.round_num, self._bank_updates
                )
                if agg_result:
                    result.global_cid  = agg_result.get("global_cid")
                    result.global_hash = agg_result.get("global_hash")
            except Exception as e:
                logger.error("Global aggregation failed: %s", e, extra={"round_num": self.config.round_num, "component": "round_coordinator"})

        result.elapsed_sec = time.time() - start
        logger.info(
            "Round complete — collected=%d submitted=%d elapsed=%.1fs global_cid=%s",
            result.collected, len(result.submitted_banks), result.elapsed_sec, result.global_cid,
            extra={"round_num": self.config.round_num, "component": "round_coordinator"}
        )
        return result
