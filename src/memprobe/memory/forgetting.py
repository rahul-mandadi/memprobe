"""Forgetting policies — ablation axis #3.

Does forgetting help or just lose information? Two policies: `none` (keep everything) and
`recency_decay` (down-weight or drop old memories by a half-life). The matrix measures whether
decay reduces staleness (old superseded values stop surfacing) without hurting task success.

`recency_weight` is implemented (pure function). Applying it inside retrieval ranking is wired
in milestone 4.
"""

from __future__ import annotations

import math

from memprobe.memory.store import MemoryRecord


def recency_weight(record: MemoryRecord, now: float, half_life: float) -> float:
    """Exponential decay weight in (0, 1]: 1.0 when fresh, 0.5 at one half-life old.

    Implemented. A ranker multiplies a record's match score by this so stale memories fade
    rather than being hard-deleted (hard deletion is a v2 variant)."""
    if half_life <= 0:
        return 1.0
    age = max(0.0, now - record.written_at)
    return math.pow(0.5, age / half_life)


# Records whose decay weight falls below this floor are dropped from retrieval entirely.
# 0.2 ~= 2.3 half-lives: with the harness's 10-day session spacing and the default 30-day
# half-life, memories older than ~7 sessions fade out — old enough to have plausibly gone
# stale, recent enough that the policy has measurable bite in an 8-session scenario.
FORGET_FLOOR = 0.2


def apply_forgetting(
    records: list[MemoryRecord], now: float, policy: str, half_life: float = 30.0
) -> list[MemoryRecord]:
    """Re-rank/filter records by forgetting policy (applied at RETRIEVAL, milestone 4).

    `none` returns the records untouched. `recency_decay` drops records whose
    recency_weight fell below FORGET_FLOOR and returns survivors freshest-first (soft decay:
    fading, not hard deletion — hard deletion is a v2 variant, forgetting.py docstring).
    Deterministic: equal weights keep input order (stable sort).
    """
    if policy in (None, "none"):
        return list(records)
    if policy != "recency_decay":
        raise ValueError(f"unknown forgetting policy: {policy!r}")
    weighted = [(recency_weight(r, now, half_life), r) for r in records]
    kept = [(w, r) for w, r in weighted if w >= FORGET_FLOOR]
    kept.sort(key=lambda wr: -wr[0])
    return [r for _, r in kept]
