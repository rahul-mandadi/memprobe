"""The three mandatory anchors that make a run interpretable (NOTES.md ADR-0006).

A run whose anchors don't hold is VOID — the harness aborts rather than reporting numbers that
can't be trusted. This module is deliberately small and strict.

Partially implemented: the validation logic (given metrics) is real; the chance-floor input
depends on metrics.chance_rate which is a milestone-1 TODO.
"""

from __future__ import annotations

from dataclasses import dataclass

from memprobe.eval.stats import Estimate, separated


@dataclass
class AnchorReport:
    ok: bool
    messages: list[str]


def validate(
    memory_off: Estimate,
    oracle: Estimate,
    placebo: Estimate,
    chance_floor: float,
    tolerance: float = 0.05,
) -> AnchorReport:
    """Check calibration, upper bound, and placebo. Returns ok=False with reasons if any fail.

    - calibration: memory_off.mean must not EXCEED chance + `tolerance` (task must not leak
      the answer without memory). One-sided on purpose: an agent that refuses instead of
      guessing sits BELOW chance, which is honest, not a failure.
    - upper_bound: oracle must be meaningfully above memory_off (else memory can't help here —
      the scenario is broken).
    - placebo: shuffled/other-user memory must not clear BOTH the empirical floor
      (CI-separated above memory_off) AND the coincidence ceiling (its whole CI above
      max(memory_off, chance) + tolerance). ADR-0013: another user's memory matches the
      right answer at the chance rate by value collision alone, while a refusing agent puts
      memory_off at 0 — so "beats memory_off" is NOT leak evidence by itself; "beats chance"
      is. Only above-coincidence placebo scores void the run.
    """
    msgs: list[str] = []
    ok = True

    if memory_off.mean > chance_floor + tolerance:
        ok = False
        msgs.append(
            f"CALIBRATION FAIL: memory_off {memory_off.mean:.3f} exceeds chance "
            f"{chance_floor:.3f}+{tolerance} — the task leaks the answer without memory."
        )

    if not (oracle.mean > memory_off.mean and separated(oracle, memory_off)):
        ok = False
        msgs.append(
            f"UPPER-BOUND FAIL: oracle {oracle} not clearly above memory_off {memory_off} — "
            "memory cannot help on these scenarios; regenerate."
        )

    placebo_ceiling = max(memory_off.mean, chance_floor) + tolerance
    if (
        placebo.mean > placebo_ceiling
        and placebo.lo > placebo_ceiling
        and separated(placebo, memory_off)
    ):
        ok = False
        msgs.append(
            f"PLACEBO FAIL: shuffled memory {placebo} beats both memory_off {memory_off} and "
            f"the chance ceiling {placebo_ceiling:.3f} — retrieval is leaking across users. "
            "Run is VOID."
        )

    # Not a FAIL, but not a pass either: the placebo's point estimate clears the coincidence
    # ceiling while its interval is too wide to say whether that is real. Found on the first
    # real-model pilot (n=2, one user): placebo 0.750 against a 0.362 ceiling, CI
    # [-2.4, 3.9], and the gate printed "anchors OK". A gate that cannot fail at small n must
    # say so rather than read as a pass.
    placebo_inconclusive = ok and placebo.mean > placebo_ceiling and placebo.lo <= placebo_ceiling
    if placebo_inconclusive:
        msgs.append(
            f"PLACEBO INCONCLUSIVE: shuffled memory {placebo} is above the coincidence ceiling "
            f"{placebo_ceiling:.3f} on its point estimate, but the interval is too wide to call. "
            "Not a pass; rerun with more users or seeds before reading the matrix."
        )
    elif ok:
        msgs.append("anchors OK: calibration, upper bound, and placebo all hold.")
    return AnchorReport(ok=ok, messages=msgs)
