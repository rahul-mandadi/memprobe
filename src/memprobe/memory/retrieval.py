"""Retrieval backends: direct structured read vs embedding similarity.

This is the PKB sequel (NOTES.md). PKB made a measured bet — no vector DB until telemetry
justified it. memprobe runs the experiment that bet implied: at these scenario scales, does
embedding retrieval actually beat a direct structured read for task success, and at what token
cost? Report whichever way it lands — "direct read held up at this scale" is a real finding,
not a failure.

`DirectRetriever` is implemented (delegates to the store's structured search). The embedding
backend needs sentence-transformers and is stubbed.
"""

from __future__ import annotations

from memprobe.memory.store import MemoryRecord, MemoryStore


class DirectRetriever:
    """Structured key-match retrieval — the baseline. Implemented."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def retrieve(self, user_id: str, query_key: str, k: int = 5) -> list[MemoryRecord]:
        return self.store.search(user_id, query_key, k=k)


class EmbeddingRetriever:
    """Semantic similarity retrieval over stored memories.

    Uses local sentence-transformers `all-MiniLM-L6-v2` (22MB, CPU, offline) by default — the
    same embedding LangGraph's Store can index with. Stubbed. TODO(milestone-3):
    embed records on write, embed the query, cosine top-k. Keep the token-cost accounting so
    the accuracy/cost comparison against DirectRetriever is fair."""

    def __init__(self, store: MemoryStore, embedder=None) -> None:
        raise NotImplementedError("TODO(milestone-3): sentence-transformers cosine top-k retrieval")
