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


def apply_forgetting(
    records: list[MemoryRecord], now: float, policy: str, half_life: float = 30.0
) -> list[MemoryRecord]:
    """Re-rank/filter records by forgetting policy. TODO(milestone-4): wire into retrieval.

    For `recency_decay`, sort by recency_weight and optionally drop below a floor. For `none`,
    return as-is."""
    raise NotImplementedError("TODO(milestone-4): apply recency_weight in retrieval ranking")
