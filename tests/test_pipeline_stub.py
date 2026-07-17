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

@pytest.mark.xfail(reason="milestone-1: chance_rate estimation", raises=NotImplementedError, strict=True)
def test_chance_rate_impl():
    from memprobe.eval.metrics import chance_rate
    from memprobe.config import ScenarioParams
    from memprobe.scenarios.generator import generate_scenario
    s = generate_scenario("u001", 0, ScenarioParams())
    chance_rate(s, s.sessions[-1].probes)


@pytest.mark.xfail(reason="milestone-2: fact extraction needs a model", raises=NotImplementedError, strict=True)
def test_extract_facts_impl():
    from memprobe.memory.semantic import extract_facts
    extract_facts([], build_model("stub"))


@pytest.mark.xfail(reason="milestone-2: LangGraph agent assembly", raises=NotImplementedError, strict=True)
def test_build_agent_impl():
    from memprobe.agent.graph import PolicyConfig, build_agent
    build_agent(PolicyConfig(), store=None, agent_model=build_model("stub"))


@pytest.mark.xfail(reason="milestone-2: LangGraph Store adapter", raises=NotImplementedError, strict=True)
def test_langgraph_store_impl():
    from memprobe.memory.store import LangGraphStore
    LangGraphStore()


@pytest.mark.xfail(reason="milestone-3: embedding retrieval", raises=NotImplementedError, strict=True)
def test_embedding_retriever_impl():
    from memprobe.memory.retrieval import EmbeddingRetriever
    EmbeddingRetriever(store=None)


@pytest.mark.xfail(reason="milestone-4: judge + agreement audit", raises=NotImplementedError, strict=True)
def test_judge_impl():
    from memprobe.eval.judge import judge_quality
    judge_quality("resp", "expected", build_model("stub"))


@pytest.mark.xfail(reason="milestone-1..5: harness orchestration", raises=NotImplementedError, strict=True)
def test_harness_run_impl():
    from memprobe.harness.run import run
    run("config/default.yaml")
