"""End-to-end smoke on the model-free path + xfail guards on the not-yet-built stubs.

The StubModel lets us prove the *shape* of the pipeline works (an oracle-style prompt yields
correct answers; a memory-off prompt yields none) before any real model exists. The xfail
tests below are the implementation worklist: each flips to a real test as its milestone lands
(xfail_strict=true means an accidentally-passing stub fails the suite loudly).
"""

import pytest

from memprobe.agent.models import StubModel, build_model
from memprobe.eval.metrics import score_probe
from memprobe.scenarios.schema import Probe


def test_stub_model_answers_from_context():
    m = build_model("stub")
    # Oracle-style: the fact is in context -> stub answers with it -> probe scores correct.
    resp = m.complete("Context: my plan_tier is pro. Q: what's my plan_tier?")
    probe = Probe(probe_id="p", session_index=0, fact_key="plan_tier", expected_value="pro")
    assert score_probe(resp, probe).correct
    assert m.usage.input_tokens > 0 and m.usage.output_tokens > 0


def test_stub_model_memory_off_has_no_answer():
    m = build_model("stub")
    resp = m.complete("Q: what's my plan_tier?")  # no fact in context
    probe = Probe(probe_id="p", session_index=0, fact_key="plan_tier", expected_value="pro")
    assert not score_probe(resp, probe).correct


def test_stub_recency_prefers_latest_value():
    m = StubModel()
    resp = m.complete("my plan_tier is free. later: my plan_tier is now pro. Q?")
    probe = Probe(probe_id="p", session_index=0, fact_key="plan_tier", expected_value="pro",
                  must_not_contain=["free"])
    r = score_probe(resp, probe)
    assert r.correct and not r.contaminated


# --- Implementation worklist: these xfail until their milestone is built ---

def test_chance_rate_is_exact_vocab_floor():
    # milestone-1 (implemented): the floor is exactly mean(1/|FACT_VOCAB[key]|) over probes.
    from memprobe.eval.metrics import chance_rate
    from memprobe.config import ScenarioParams
    from memprobe.scenarios.generator import FACT_VOCAB, generate_scenario
    s = generate_scenario("u001", 0, ScenarioParams())
    probes = s.sessions[-1].probes
    got = chance_rate(s, probes)
    want = sum(1.0 / len(FACT_VOCAB[p.fact_key]) for p in probes) / len(probes)
    assert abs(got - want) < 1e-12
    assert 0.0 < got < 1.0


def test_extract_facts_confidence_is_a_real_signal():
    # milestone-2 (implemented): three evidence tiers -> tau has something to separate.
    from memprobe.memory.semantic import (
        CONF_OFF_VOCAB, CONF_UPDATE, CONF_VOCAB_ASSERT, extract_facts,
    )
    from memprobe.scenarios.schema import Turn

    m = build_model("stub")
    assert extract_facts([], m) == []
    turns = [
        Turn(speaker="user", text="my plan_tier is pro"),
        Turn(speaker="user", text="my noise_billing_cycle is annual"),
        Turn(speaker="user", text="actually my timezone is now pacific"),
    ]
    cands = {c.key: c for c in extract_facts(turns, m)}
    assert cands["plan_tier"].confidence == CONF_VOCAB_ASSERT
    assert cands["noise_billing_cycle"].confidence == CONF_OFF_VOCAB
    assert cands["timezone"].confidence == CONF_UPDATE
    assert len({c.confidence for c in cands.values()}) == 3


def test_build_agent_runs_the_full_graph():
    # milestone-2 (implemented): a fact written in session 0 answers a probe in session 1.
    from memprobe.agent.graph import PolicyConfig, build_agent
    from memprobe.memory.store import DictStore
    from memprobe.scenarios.schema import Session, Turn

    agent = build_agent(PolicyConfig(use_semantic=True), DictStore(), build_model("stub"))
    s0 = Session(user_id="a", index=0,
                 turns=[Turn(speaker="user", text="my plan_tier is pro")], probes=[])
    assert agent.run_session("a", s0) == []
    probe = Probe(probe_id="p", session_index=1, fact_key="plan_tier", expected_value="pro")
    s1 = Session(user_id="a", index=1, turns=[], probes=[probe])
    (resp,) = agent.run_session("a", s1)
    assert score_probe(resp["response"], probe).correct


def test_langgraph_store_adapter():
    # milestone-2 (implemented): append semantics, insertion order, recency search, isolation.
    from memprobe.memory.store import LangGraphStore, MemoryRecord

    s = LangGraphStore()
    s.put(MemoryRecord(key="plan_tier", value="free", user_id="a", source_session=0, written_at=0.0))
    s.put(MemoryRecord(key="plan_tier", value="pro", user_id="a", source_session=2, written_at=2.0))
    s.put(MemoryRecord(key="plan_tier", value="team", user_id="b", source_session=0, written_at=0.0))
    assert [r.value for r in s.all("a")] == ["free", "pro"]
    assert [r.value for r in s.get("a", "plan_tier")] == ["free", "pro"]
    assert [r.value for r in s.search("a", "plan_tier", k=1)] == ["pro"]
    assert [r.value for r in s.all("b")] == ["team"]


def test_embedding_retriever_cosine_topk():
    # milestone-3 (implemented): similarity search finds the right fact record for a
    # natural-language query, hermetically (deterministic hashing embedder).
    from memprobe.memory.retrieval import EmbeddingRetriever, HashingEmbedder
    from memprobe.memory.store import DictStore, MemoryRecord

    store = DictStore()
    store.put(MemoryRecord(key="plan_tier", value="pro", user_id="a", source_session=0, written_at=0.0))
    store.put(MemoryRecord(key="timezone", value="eastern", user_id="a", source_session=1, written_at=1.0))
    ret = EmbeddingRetriever(store, embedder=HashingEmbedder())
    top = ret.retrieve("a", "what's my plan tier on file?", k=1)
    assert [r.key for r in top] == ["plan_tier"]
    assert ret.embed_tokens > 0  # the cost meter runs


def test_judge_runs_with_stub_and_audit_gate_exists():
    # milestone-4 (implemented): the judge runs deterministically with the stub (flagged as
    # a fallback, never a model judgment) and the audit machinery computes kappa.
    from memprobe.eval.judge import audit_agreement, judge_quality

    score = judge_quality("plan_tier: pro", "pro", build_model("stub"))
    assert score.quality == 1.0 and score.rationale.startswith("[fallback slot-check]")
    audit = audit_agreement({"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0})
    assert audit.n_sampled == 2 and audit.agreement == 1.0 and audit.cohen_kappa == 1.0


@pytest.mark.xfail(reason="milestone-1..5: harness orchestration", raises=NotImplementedError, strict=True)
def test_harness_run_impl():
    from memprobe.harness.run import run
    run("config/default.yaml")
