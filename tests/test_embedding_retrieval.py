"""Embedding retrieval (milestone 3): deterministic hashing backend, cosine top-k, fairness
accounting. All hermetic — the sentence-transformers path is the same code with a different
encoder injected."""

import numpy as np
import pytest

from memprobe.memory.retrieval import (
    EmbeddingRetriever,
    HashingEmbedder,
    build_embedder,
    _tokens,
)
from memprobe.memory.store import DictStore, LangGraphStore, MemoryRecord


def _rec(key, value, user="a", session=0, at=0.0):
    return MemoryRecord(key=key, value=value, user_id=user, source_session=session, written_at=at)


def test_tokenizer_splits_underscores():
    # 'plan_tier' must meet the query surface 'plan tier' at the token level.
    assert _tokens("my plan_tier is pro") == ["my", "plan", "tier", "is", "pro"]


def test_hashing_embedder_is_deterministic_across_instances():
    a = HashingEmbedder().encode(["my plan tier is pro"])
    b = HashingEmbedder().encode(["my plan tier is pro"])
    assert np.allclose(a, b)
    assert abs(float(np.linalg.norm(a[0])) - 1.0) < 1e-6  # unit vectors


def test_hashing_embedder_ranks_related_text_closer():
    emb = HashingEmbedder()
    q, plan, tz = emb.encode([
        "what's my plan tier on file?",
        "my plan tier is pro",
        "my timezone is eastern",
    ])
    assert float(np.dot(q, plan)) > float(np.dot(q, tz))


def test_build_embedder_factory():
    assert isinstance(build_embedder("hash"), HashingEmbedder)
    assert build_embedder("hash:512").dim == 512
    from memprobe.memory.retrieval import SentenceTransformerEmbedder

    st = build_embedder("local:all-MiniLM-L6-v2")
    assert isinstance(st, SentenceTransformerEmbedder)  # constructed lazily, no import yet
    with pytest.raises(ValueError):
        build_embedder("faiss:whatever")


@pytest.mark.parametrize("backend", [DictStore, LangGraphStore])
def test_retrieve_topk_finds_the_probed_fact(backend):
    store = backend()
    store.put(_rec("plan_tier", "free", at=0.0))
    store.put(_rec("timezone", "eastern", at=1.0))
    store.put(_rec("plan_tier", "pro", at=2.0))
    store.put(_rec("primary_device", "ios", at=3.0))
    ret = EmbeddingRetriever(store, embedder=HashingEmbedder())
    top = ret.retrieve("a", "what's my plan tier on file?", k=2)
    assert {r.key for r in top} == {"plan_tier"}
    # Equal-similarity duplicates tie-break by recency: 'pro' (newest) first.
    assert [r.value for r in top] == ["pro", "free"]


def test_cross_user_isolation():
    store = DictStore()
    store.put(_rec("plan_tier", "pro", user="a"))
    store.put(_rec("plan_tier", "enterprise", user="b"))
    ret = EmbeddingRetriever(store, embedder=HashingEmbedder())
    assert [r.value for r in ret.retrieve("a", "plan tier", k=5)] == ["pro"]
    assert ret.retrieve("nobody", "plan tier", k=5) == []


def test_episode_records_stay_on_their_own_channel():
    store = DictStore()
    store.put(_rec("plan_tier", "pro"))
    store.put(_rec("episode:000", "plan_tier: pro; timezone: eastern", at=1.0))
    ret = EmbeddingRetriever(store, embedder=HashingEmbedder())
    assert [r.key for r in ret.retrieve("a", "plan tier", k=5)] == ["plan_tier"]


def test_cost_meter_counts_encodes_once_per_text():
    store = DictStore()
    store.put(_rec("plan_tier", "pro"))
    store.put(_rec("timezone", "eastern", at=1.0))
    ret = EmbeddingRetriever(store, embedder=HashingEmbedder())
    ret.retrieve("a", "what's my plan tier on file?", k=2)
    after_first = ret.embed_tokens
    assert after_first > 0
    ret.retrieve("a", "what's my plan tier on file?", k=2)
    assert ret.embed_tokens == after_first  # records + query cached; no new encoder work
    ret.retrieve("a", "which timezone am I in?", k=2)
    assert ret.embed_tokens > after_first  # a new query is real encoder work


def test_agent_graph_embedding_policy_end_to_end():
    from memprobe.agent.graph import PolicyConfig, build_agent
    from memprobe.agent.models import StubModel
    from memprobe.eval.metrics import score_probe
    from memprobe.scenarios.schema import Probe, Session, Turn

    store = DictStore()
    policy = PolicyConfig(use_semantic=True, retrieval="embedding")
    agent = build_agent(policy, store, StubModel(), embedder=HashingEmbedder())
    s0 = Session(user_id="a", index=0,
                 turns=[Turn(speaker="user", text="my plan_tier is pro"),
                        Turn(speaker="user", text="my timezone is eastern")], probes=[])
    agent.run_session("a", s0)
    probe = Probe(probe_id="p", session_index=1, fact_key="plan_tier", expected_value="pro")
    (resp,) = agent.run_session("a", Session(user_id="a", index=1, turns=[], probes=[probe]))
    assert score_probe(resp["response"], probe).correct
