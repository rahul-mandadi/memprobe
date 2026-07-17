"""Deterministic metrics — the credibility core. Scored on hand-built fixtures."""

from memprobe.eval.metrics import _contains_value, aggregate, score_probe
from memprobe.scenarios.schema import Probe


def _probe(expected, avoid=None):
    return Probe(probe_id="p1", session_index=0, fact_key="plan_tier",
                expected_value=expected, must_not_contain=avoid or [])


def test_correct_answer():
    r = score_probe("your plan tier is pro", _probe("pro"))
    assert r.correct and not r.contaminated and not r.stale


def test_word_boundary_prevents_false_match():
    # "pro" must not match inside "problem"; the agent didn't actually state the value.
    r = score_probe("no problem, let me check", _probe("pro"))
    assert not r.correct


def test_contamination_detected():
    # surfaces another user's value ("enterprise") -> contaminated, and since not the expected
    # current value, also flagged stale-or-wrong via the shared must_not_contain channel.
    r = score_probe("you're on enterprise", _probe("pro", avoid=["enterprise"]))
    assert r.contaminated and not r.correct


def test_correct_despite_listing_avoided_only_counts_contamination_when_present():
    r = score_probe("your plan tier is pro", _probe("pro", avoid=["free"]))
    assert r.correct and not r.contaminated


def test_aggregate_rates():
    results = [
        score_probe("pro", _probe("pro")),
        score_probe("free", _probe("pro", avoid=["free"])),  # wrong + contaminated
        score_probe("pro", _probe("pro")),
    ]
    m = aggregate(results)
    assert m.n_probes == 3
    assert abs(m.task_success - 2 / 3) < 1e-9
    assert abs(m.contamination_rate - 1 / 3) < 1e-9


def test_contains_value_normalizes_punctuation_and_case():
    assert _contains_value("Plan: PRO.", "pro")
    assert not _contains_value("", "pro")
