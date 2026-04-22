"""
fl-layer/resilience/deadline_collect.py
Deadline-aware submission collector with injectable collect_fn.

Changes vs. CCFD-FL-layer/src/resilience/deadline_collect.py:
  - Removed Docker filesystem polling (no round_dir, no state_dict loading).
  - collect_fn is a callable that returns whatever is currently available.
  - This makes the function fully testable without Docker or filesystem.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, List

logger = logging.getLogger(__name__)


def wait_for_submissions(
    expected_count: int,
    collect_fn: Callable[[], List[Any]],
    deadline_sec: float = 25.0,
    poll_interval: float = 0.5,
) -> List[Any]:
    """
    Wait up to deadline_sec for collect_fn to return expected_count items.

    Args:
        expected_count : target number of submissions (e.g. number of branches)
        collect_fn     : zero-argument callable that returns the list of currently
                         available submissions (called repeatedly until deadline)
        deadline_sec   : maximum wall-clock seconds to wait
        poll_interval  : seconds between collect_fn calls

    Returns:
        Whatever collect_fn returned at the moment the deadline was hit or
        expected_count was satisfied.
    """
    start = time.time()
    logger.debug(
        "wait_for_submissions: expecting %d submissions, deadline %.1fs",
        expected_count, deadline_sec,
    )

    while True:
        current = collect_fn()
        if len(current) >= expected_count:
            logger.debug("All %d submissions received early (%.1fs elapsed).",
                         expected_count, time.time() - start)
            return current

        elapsed = time.time() - start
        if elapsed >= deadline_sec:
            logger.warning(
                "Deadline reached (%.1fs). Got %d/%d submissions.",
                elapsed, len(current), expected_count,
            )
            return current

        time.sleep(poll_interval)
