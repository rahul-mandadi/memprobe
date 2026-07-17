"""Episodic memory: per-session summaries the agent can recall in later sessions.

Simplest useful memory type and a clean baseline column (episodic_only). A session summary is
written at session close; retrieval brings back summaries of prior sessions. Contrast with
semantic memory, which extracts atomic facts — the matrix measures whether atomic facts beat
whole-summary recall for task success and at what token cost.

Summarization needs a model -> stubbed. The store write path reuses store.MemoryRecord.
"""

from __future__ import annotations

from memprobe.memory.store import MemoryStore


def summarize_session(turns, model) -> str:
    """Produce a short episode summary of one session. TODO(milestone-2)."""
    raise NotImplementedError("TODO(milestone-2): session summarization via model")


def write_episode(store: MemoryStore, user_id: str, session_index: int, summary: str, now: float) -> None:
    """Persist an episode summary as a memory record. TODO(milestone-2): wire once summarize lands."""
    raise NotImplementedError("TODO(milestone-2): store.put an episodic MemoryRecord")
