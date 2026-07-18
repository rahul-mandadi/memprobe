"""Memory store abstraction.

Design (NOTES.md ADR-0003): the *substrate* is LangGraph's `Store`
(`langgraph.store.memory.InMemoryStore`, `put(namespace, key, value)` / `search(...)`). We do
NOT reinvent the datastore. What memprobe owns is the thin layer on top: per-user namespacing,
provenance/timestamp stamping, and the policy hooks (write-gate, forgetting) that the ablation
sweeps.

To keep the policy logic hermetically testable (no langgraph install, no network), the store
is a Protocol with two backends:
  - `DictStore`   — reference in-process backend; used by the test suite and model-free runs.
  - `LangGraphStore` — thin adapter over InMemoryStore for real runs (stubbed).
Both present the same namespaced put/get/search surface, so every downstream policy module is
backend-agnostic and the direct-vs-embedding comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class MemoryRecord:
    """One stored memory. Provenance + timestamp are first-class so staleness/forgetting and
    the citation-integrity idea (carried over from the PKB project) are measurable."""

    key: str
    value: str
    user_id: str
    source_session: int        # provenance: which session produced this memory
    written_at: float          # logical timestamp (session index or day), for forgetting
    confidence: float = 1.0    # extractor confidence; the write-gate compares against tau
    meta: dict = field(default_factory=dict)


class MemoryStore(Protocol):
    """Namespaced by user_id. Cross-user isolation is the invariant test_store.py enforces."""

    def put(self, record: MemoryRecord) -> None: ...
    def get(self, user_id: str, key: str) -> list[MemoryRecord]: ...
    def all(self, user_id: str) -> list[MemoryRecord]: ...
    def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryRecord]: ...


class DictStore:
    """Reference backend: dict keyed by (user_id) -> list[MemoryRecord].

    Implemented. `search` here is the DIRECT-read baseline (structured key match); the
    embedding backend lives in retrieval.py and wraps a store to add similarity search. Cross
    user isolation is structural: a user_id only ever sees its own bucket.
    """

    def __init__(self) -> None:
        self._by_user: dict[str, list[MemoryRecord]] = {}

    def put(self, record: MemoryRecord) -> None:
        self._by_user.setdefault(record.user_id, []).append(record)

    def get(self, user_id: str, key: str) -> list[MemoryRecord]:
        return [r for r in self._by_user.get(user_id, []) if r.key == key]

    def all(self, user_id: str) -> list[MemoryRecord]:
        return list(self._by_user.get(user_id, []))

    def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryRecord]:
        # Direct baseline: exact/substring key match, most-recent first. No semantics.
        hits = [r for r in self._by_user.get(user_id, []) if r.key in query or query in r.key]
        return sorted(hits, key=lambda r: r.written_at, reverse=True)[:k]


class LangGraphStore:
    """Adapter over langgraph.store.memory.InMemoryStore (the production substrate, ADR-0003).

    LangGraph's Store owns storage and namespacing — namespaces map to its namespace tuples:
    ("memprobe", user_id). Two adapter decisions worth defending:

    - Item keys are "{record.key}@{seq}" because LangGraph's put() REPLACES by key, while the
      lab needs APPEND semantics: a superseded value must stay observable or staleness and
      forgetting have nothing to measure. The monotone seq also preserves insertion order.
    - `search` re-implements the SAME transparent direct-read ranking as DictStore (substring
      key match, most-recent first) rather than delegating to the substrate's text search.
      Both backends must rank identically so the direct-vs-embedding ablation measures the
      retrieval *policy*, not two vendors' ranking implementations. Parity is test-enforced
      (test_langgraph_store.py).
    """

    NAMESPACE_ROOT = "memprobe"

    def __init__(self, index: dict | None = None) -> None:
        # `index` configures LangGraph semantic search (embeddings). memprobe's own
        # EmbeddingRetriever (retrieval.py) is the measured embedding path; the index
        # passthrough exists so the substrate's native indexing stays reachable.
        from langgraph.store.memory import InMemoryStore

        self._store = InMemoryStore(index=index) if index is not None else InMemoryStore()
        self._seq = 0

    def _ns(self, user_id: str) -> tuple[str, str]:
        return (self.NAMESPACE_ROOT, user_id)

    def put(self, record: MemoryRecord) -> None:
        from dataclasses import asdict

        self._seq += 1
        self._store.put(self._ns(record.user_id), f"{record.key}@{self._seq:08d}", asdict(record))

    def get(self, user_id: str, key: str) -> list[MemoryRecord]:
        return [r for r in self.all(user_id) if r.key == key]

    def all(self, user_id: str) -> list[MemoryRecord]:
        items = self._store.search(self._ns(user_id), limit=100_000)
        # Insertion order via the seq suffix — the substrate does not guarantee order.
        items.sort(key=lambda it: int(it.key.rsplit("@", 1)[1]))
        return [MemoryRecord(**it.value) for it in items]

    def search(self, user_id: str, query: str, k: int = 5) -> list[MemoryRecord]:
        # Direct baseline: identical semantics to DictStore.search (parity is the contract).
        hits = [r for r in self.all(user_id) if r.key in query or query in r.key]
        return sorted(hits, key=lambda r: r.written_at, reverse=True)[:k]
