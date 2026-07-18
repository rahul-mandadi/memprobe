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


def _tokens(text: str) -> list[str]:
    """Shared tokenizer for hashing + cost accounting: lowercase word tokens, underscores
    split (so 'plan_tier' meets 'plan tier' — the query surface probes actually use)."""
    import re

    return re.findall(r"[a-z0-9]+", text.replace("_", " ").lower())


class HashingEmbedder:
    """Deterministic feature-hashing embedder — the hermetic embedding backend (ADR-0011).

    Classic feature hashing (Weinberger et al. 2009): md5(token) -> bucket, count,
    l2-normalize. md5, not Python's hash(), so vectors are stable across processes and
    platforms (PYTHONHASHSEED would silently break reproducibility). No dependencies, no
    model download — this is what lets the embedding-retrieval column run in CI and in the
    default hermetic matrix. It measures the retrieval POLICY (similarity top-k vs direct
    key match); `local:all-MiniLM-L6-v2` is the drop-in real encoder for opt-in runs.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def encode(self, texts: list[str]):
        import hashlib

        import numpy as np

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokens(text):
                bucket = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "big") % self.dim
                out[i, bucket] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return out / norms


class SentenceTransformerEmbedder:
    """Real local encoder (sentence-transformers, 22MB, CPU, offline once downloaded).

    Lazy import: the [local] extra is opt-in; the hermetic core never needs it.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "SentenceTransformerEmbedder needs the [local] extra: "
                    "pip install -e '.[local]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode(list(texts), normalize_embeddings=True)


def build_embedder(spec: str):
    """Factory: 'hash' | 'hash:<dim>' | 'local:<sentence-transformers model>'."""
    provider, _, arg = spec.partition(":")
    if provider == "hash":
        return HashingEmbedder(dim=int(arg) if arg else 256)
    if provider == "local":
        return SentenceTransformerEmbedder(arg or "all-MiniLM-L6-v2")
    raise ValueError(f"unknown embedder spec: {spec!r}")


class EmbeddingRetriever:
    """Semantic similarity retrieval over stored memories: cosine top-k (milestone 3).

    Design decisions that keep the direct-vs-embedding comparison fair (ADR-0011):
    - Records render into the same natural surface the responder sees ("my plan tier is
      pro") and the QUERY is the natural probe question — similarity search gets exactly the
      inputs a production system would have, no more (the direct read gets the structured
      key, which is what makes it "direct").
    - Episode records are EXCLUDED here: episodic recall has its own recency channel in the
      agent, so the retrieval axis changes how fact records are found and nothing else (one
      variable per ablation).
    - Token/cost accounting: every text actually ENCODED is metered in `embed_tokens`.
      Records and queries are cached by exact text (an incrementally built index plus a
      query cache — both standard practice), so the meter counts real encoder work.
      DirectRetriever's embedding cost is structurally zero, and the report's cost column
      keeps that comparison honest.
    """

    def __init__(self, store: MemoryStore, embedder=None) -> None:
        self.store = store
        self._embedder = embedder
        self._cache: dict[str, object] = {}
        self.embed_tokens = 0
        self.embed_calls = 0

    @property
    def embedder(self):
        if self._embedder is None:
            # Honor the documented default (MiniLM) but resolve lazily so construction —
            # and every hermetic code path that never retrieves — needs no [local] extra.
            self._embedder = build_embedder("local:all-MiniLM-L6-v2")
        return self._embedder

    @staticmethod
    def _record_text(record: MemoryRecord) -> str:
        return f"my {record.key.replace('_', ' ')} is {record.value}"

    def _encode(self, texts: list[str]):
        misses = [t for t in texts if t not in self._cache]
        if misses:
            vecs = self.embedder.encode(misses)
            for t, v in zip(misses, vecs):
                self._cache[t] = v
            self.embed_tokens += sum(len(_tokens(t)) for t in misses)
            self.embed_calls += 1
        return [self._cache[t] for t in texts]

    def retrieve(self, user_id: str, query: str, k: int = 5) -> list[MemoryRecord]:
        import numpy as np

        from memprobe.memory.episodic import EPISODE_KEY_PREFIX

        records = [
            r for r in self.store.all(user_id)
            if not r.key.startswith(EPISODE_KEY_PREFIX + ":")
        ]
        if not records:
            return []
        # Queries are cached under a namespaced key so a query string can never collide
        # with a record text and silently reuse its vector.
        query_key = f"query::{query}"
        if query_key in self._cache:
            qvec = self._cache[query_key]
        else:
            (qvec,) = self.embedder.encode([query])
            self._cache[query_key] = qvec
            self.embed_tokens += len(_tokens(query))
            self.embed_calls += 1
        rvecs = self._encode([self._record_text(r) for r in records])
        scores = [float(np.dot(qvec, rv)) for rv in rvecs]
        ranked = sorted(
            range(len(records)),
            key=lambda i: (-scores[i], -records[i].written_at, i),
        )
        return [records[i] for i in ranked[:k]]
