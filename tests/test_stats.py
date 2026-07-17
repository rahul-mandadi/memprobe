"""Confidence intervals and CI-separation."""

from memprobe.eval.stats import mean_ci, separated


def test_mean_ci_basic():
    est = mean_ci([0.4, 0.5, 0.6], confidence=0.95)
    assert abs(est.mean - 0.5) < 1e-9
    assert est.lo < est.mean < est.hi
    assert est.n == 3


def test_zero_width_for_singleton():
    est = mean_ci([0.7])
    assert est.lo == est.hi == est.mean == 0.7


def test_empty_is_degenerate():
    est = mean_ci([])
    assert est.n == 0 and est.mean == 0.0


def test_separated_true_when_disjoint():
    lo = mean_ci([0.10, 0.11, 0.09])
    hi = mean_ci([0.90, 0.91, 0.89])
    assert separated(lo, hi)


def test_not_separated_when_overlapping():
    a = mean_ci([0.50, 0.55, 0.45])
    b = mean_ci([0.52, 0.48, 0.53])
    assert not separated(a, b)
