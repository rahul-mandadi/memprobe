"""Anchor validation — calibration / upper-bound / placebo gates (ADR-0006)."""

from memprobe.eval.anchors import validate
from memprobe.eval.stats import mean_ci


def _est(vals):
    return mean_ci(vals)


def test_all_anchors_hold():
    memory_off = _est([0.24, 0.26, 0.25])   # ~ chance 0.25
    oracle = _est([0.95, 0.96, 0.94])
    placebo = _est([0.24, 0.25, 0.26])
    rep = validate(memory_off, oracle, placebo, chance_floor=0.25)
    assert rep.ok, rep.messages


def test_calibration_fail_when_memory_off_beats_chance():
    memory_off = _est([0.60, 0.62, 0.61])   # way above chance -> task leaks
    oracle = _est([0.95, 0.96, 0.94])
    placebo = _est([0.24, 0.25, 0.26])
    rep = validate(memory_off, oracle, placebo, chance_floor=0.25)
    assert not rep.ok
    assert any("CALIBRATION" in m for m in rep.messages)


def test_upper_bound_fail_when_oracle_not_above_floor():
    memory_off = _est([0.24, 0.26, 0.25])
    oracle = _est([0.25, 0.26, 0.24])       # oracle no better than memory_off
    placebo = _est([0.24, 0.25, 0.26])
    rep = validate(memory_off, oracle, placebo, chance_floor=0.25)
    assert not rep.ok
    assert any("UPPER-BOUND" in m for m in rep.messages)


def test_placebo_fail_when_shuffled_beats_floor():
    memory_off = _est([0.24, 0.26, 0.25])
    oracle = _est([0.95, 0.96, 0.94])
    placebo = _est([0.70, 0.72, 0.71])      # other-user memory helping -> leak
    rep = validate(memory_off, oracle, placebo, chance_floor=0.25)
    assert not rep.ok
    assert any("PLACEBO" in m for m in rep.messages)
