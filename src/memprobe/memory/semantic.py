"""Semantic memory: extract user facts from turns and write them under a confidence gate.

The write-gate is the star ablation axis (NOTES.md ADR-0004): a candidate fact is written only
if extraction confidence >= tau. Sweeping tau (0.7 vs 0.9 in the default matrix) turns a vague
"we gate risky writes" feature into a measured curve: high tau -> fewer false memories
(less contamination) but more misses (lower task success). That trade-off IS the result.

`apply_write_gate` is implemented (pure policy). Extraction needs a model and is stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass

from memprobe.memory.store import MemoryRecord, MemoryStore


@dataclass
class CandidateFact:
    key: str
    value: str
    confidence: float
    source_session: int


def apply_write_gate(
    store: MemoryStore,
    user_id: str,
    candidates: list[CandidateFact],
    tau: float,
    now: float,
) -> int:
    """Write only candidates with confidence >= tau. Returns count written. Implemented.

    This is the whole gate: transparent, one comparison. A later value for the same key
    supersedes earlier ones at read time via provenance/timestamp (retrieval + forgetting
    handle recency), so the gate's only job is the accept/reject decision.
    """
    written = 0
    for c in candidates:
        if c.confidence >= tau:
            store.put(MemoryRecord(
                key=c.key, value=c.value, user_id=user_id,
                source_session=c.source_session, written_at=now, confidence=c.confidence,
            ))
            written += 1
    return written


def extract_facts(turns, model) -> list[CandidateFact]:
    """Extract candidate (key, value, confidence) facts from conversation turns via the model.

    Confidence is the extractor's own calibrated certainty — it must be a real signal, not a
    constant, or the write-gate ablation is meaningless. TODO(milestone-2)."""
    raise NotImplementedError("TODO(milestone-2): model-based fact extraction with confidence")
