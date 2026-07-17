"""Confidence intervals and config comparison.

Fully implemented. A single-run point number is not a result (NOTES.md ADR-0006): every
reported metric is a mean over per-user, per-seed observations with a 95% CI. Two configs
"differ" only if we can say so with the intervals — this module is what stops the README from
overclaiming a 2-point difference that's inside the noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats


@dataclass
class Estimate:
    mean: float
    lo: float          # lower CI bound
    hi: float          # upper CI bound
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def mean_ci(values: list[float], confidence: float = 0.95) -> Estimate:
    """Mean with a t-interval. Degenerate cases (n<2) return a zero-width interval."""
    n = len(values)
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, 0)
    mean = sum(values) / n
    if n < 2:
        return Estimate(mean, mean, mean, n)
    sem = stats.tstd(values) / math.sqrt(n)
    half = sem * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return Estimate(mean, mean - half, mean + half, n)


def separated(a: Estimate, b: Estimate) -> bool:
    """True if two estimates' CIs do not overlap (a conservative 'they differ' test)."""
    return a.hi < b.lo or b.hi < a.lo
