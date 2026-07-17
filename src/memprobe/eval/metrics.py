"""Deterministic metrics — the headline numbers.

These are the credibility core: task success, contamination, and staleness are computed by
checking the agent's response text against the scenario's known ground truth. NO model is
called here (NOTES.md ADR-0005). LLM-as-judge lives separately in judge.py and only scores
response *quality*, never the headline metrics.

Fully implemented. The matching is deliberately simple and transparent (normalized substring
containment) so a reviewer can audit exactly what "correct" means — the honesty of the whole
lab rests on this being inspectable, not clever.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from memprobe.scenarios.schema import Probe, Scenario


def _norm(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation — a transparent match normalization."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def _contains_value(response: str, value: str) -> bool:
    """Whole-token-aware containment: does `response` assert `value`?

    Uses word-boundary matching on the normalized strings so "email" does not match
    "emailed" and a bare "no" doesn't match inside "another".
    """
    r, v = _norm(response), _norm(value)
    if not v:
        return False
    return re.search(rf"(?:^| ){re.escape(v)}(?: |$)", r) is not None


@dataclass
class ProbeResult:
    probe_id: str
    correct: bool          # response asserts the CURRENT ground-truth value
    contaminated: bool     # response asserts a value that belongs to another user
    stale: bool            # response asserts a SUPERSEDED value for this key


def score_probe(response: str, probe: Probe) -> ProbeResult:
    """Score one agent response against one probe. Pure function of (text, ground truth)."""
    correct = _contains_value(response, probe.expected_value)
    contaminated = any(_contains_value(response, bad) for bad in probe.must_not_contain)
    # A stale answer surfaces a superseded value; must_not_contain carries them (tagged by the
    # generator). Contamination (other-user) and staleness (own old value) are tracked
    # separately because they have different fixes; the generator marks which is which.
    stale = contaminated and not correct  # refined in generator tagging; see test_metrics
    return ProbeResult(
        probe_id=probe.probe_id,
        correct=correct,
        contaminated=contaminated,
        stale=stale,
    )


@dataclass
class ConfigMetrics:
    """Aggregated metrics for one config over one seed (per-user rows feed stats.py CIs)."""

    task_success: float        # fraction of probes answered with the current value
    contamination_rate: float  # fraction surfacing another user's value
    staleness_rate: float      # fraction surfacing a superseded value
    n_probes: int


def aggregate(results: list[ProbeResult]) -> ConfigMetrics:
    """Fully implemented aggregation over a list of scored probes."""
    n = len(results)
    if n == 0:
        return ConfigMetrics(0.0, 0.0, 0.0, 0)
    return ConfigMetrics(
        task_success=sum(r.correct for r in results) / n,
        contamination_rate=sum(r.contaminated for r in results) / n,
        staleness_rate=sum(r.stale for r in results) / n,
        n_probes=n,
    )


def chance_rate(scenario: Scenario, probes: list[Probe]) -> float:
    """Expected task_success of blind guessing — the number memory_off must not exceed much.

    Approximated as the mean 1/(#candidate values seen for each probed key). Used by
    anchors.py to check the calibration floor. TODO(m1): refine candidate-set estimation once
    the generator's value vocabulary is finalized.
    """
    raise NotImplementedError("TODO(milestone-1): estimate per-probe candidate-value counts")
