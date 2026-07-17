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

    - calibration: memory_off.mean must be within `tolerance` of the chance floor (task must
      not leak the answer without memory).
    - upper_bound: oracle must be meaningfully above memory_off (else memory can't help here —
      the scenario is broken).
    - placebo: shuffled/other-user memory must NOT be separated-above memory_off (if it beats
      the floor, retrieval is leaking across users).
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

    if placebo.mean > memory_off.mean and separated(placebo, memory_off):
        ok = False
        msgs.append(
            f"PLACEBO FAIL: shuffled memory {placebo} beats memory_off {memory_off} — "
            "retrieval is leaking across users. Run is VOID."
        )

    if ok:
        msgs.append("anchors OK: calibration, upper bound, and placebo all hold.")
    return AnchorReport(ok=ok, messages=msgs)
