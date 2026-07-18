"""Episodic memory: per-session summaries the agent can recall in later sessions.

Simplest useful memory type and a clean baseline column (episodic_only). A session summary is
written at session close; retrieval brings back summaries of prior sessions. Contrast with
semantic memory, which extracts atomic facts — the matrix measures whether atomic facts beat
whole-summary recall for task success and at what token cost.

Summarization needs a model -> stubbed. The store write path reuses store.MemoryRecord.
"""

from __future__ import annotations

from memprobe.memory.store import MemoryRecord, MemoryStore

# Episode records are keyed with this prefix so the direct (key-match) retriever never
# confuses them with semantic fact records; the agent's retrieve node recalls episodes by
# recency instead (episodic memory is "remember past sessions", not a keyed lookup).
EPISODE_KEY_PREFIX = "episode"

# The summary format is a CONTRACT: a compact structured recap of 'key: value' pairs. The
# respond-time context builder re-renders these pairs into the canonical fact surface
# (ADR-0009), which is what makes episodic memory comparable to semantic memory without a
# format confound. The deterministic StubModel emits exactly this shape by construction; a
# real model is instructed to.
_SUMMARY_INSTRUCTION = (
    "Summarize this support session as a compact recap of the user's stated facts, one "
    "'<key>: <value>' pair per fact, latest value only, semicolon-separated. No other text.\n\n"
    "Session:\n"
)


def summarize_session(turns, model) -> str:
    """Produce a short structured episode summary of one session (empty if nothing to keep).

    Note what episodic memory does NOT get: no write gate, no per-fact confidence — the whole
    session's surface (distractors included) is recapped wholesale. That asymmetry versus
    gated semantic extraction is the episodic-vs-semantic ablation contrast, on purpose.
    """
    user_texts = [t.text for t in turns if getattr(t, "speaker", "user") == "user"]
    if not user_texts:
        return ""
    return model.complete(_SUMMARY_INSTRUCTION + "\n".join(user_texts)).strip()


def write_episode(store: MemoryStore, user_id: str, session_index: int, summary: str, now: float) -> None:
    """Persist an episode summary as a memory record (no-op for an empty summary)."""
    if not summary:
        return
    store.put(MemoryRecord(
        key=f"{EPISODE_KEY_PREFIX}:{session_index:03d}",
        value=summary,
        user_id=user_id,
        source_session=session_index,
        written_at=now,
        confidence=1.0,  # episodes are unconditional writes — that IS the policy contrast
        meta={"type": "episodic"},
    ))
