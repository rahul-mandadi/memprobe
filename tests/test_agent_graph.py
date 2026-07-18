"""One graph, different knobs: each anchor/policy mode behaves as designed (ADR-0006/0009/0010)."""

from memprobe.agent.graph import PolicyConfig, build_agent
from memprobe.agent.models import StubModel
from memprobe.eval.metrics import score_probe
from memprobe.memory.store import DictStore, LangGraphStore
from memprobe.scenarios.schema import Probe, Session, Turn


def _session(user_id, index, texts=(), probes=()):
    return Session(
        user_id=user_id, index=index,
        turns=[Turn(speaker="user", text=t) for t in texts],
        probes=list(probes),
    )


def _probe(key="plan_tier", expected="pro", avoid=(), session=1):
    return Probe(probe_id=f"{key}-p", session_index=session, fact_key=key,
                 expected_value=expected, must_not_contain=list(avoid))


def test_memory_off_answers_nothing():
    agent = build_agent(PolicyConfig(memory_off=True), None, StubModel())
    p = _probe()
    (resp,) = agent.run_session("a", _session("a", 1, probes=[p]))
    assert not score_probe(resp["response"], p).correct


def test_oracle_injects_ground_truth_and_is_correct():
    agent = build_agent(PolicyConfig(oracle=True), None, StubModel())
    p = _probe()
    (resp,) = agent.run_session(
        "a", _session("a", 1, probes=[p]), oracle_facts={"plan_tier": "pro"}
    )
    assert score_probe(resp["response"], p).correct


def test_semantic_memory_recalls_across_sessions_on_langgraph_backend():
    # The production substrate path: same behavior as DictStore (parity by construction).
    store = LangGraphStore()
    agent = build_agent(PolicyConfig(use_semantic=True), store, StubModel())
    agent.run_session("a", _session("a", 0, texts=["my plan_tier is pro"]))
    p = _probe()
    (resp,) = agent.run_session("a", _session("a", 1, probes=[p]))
    assert score_probe(resp["response"], p).correct


def test_scramble_reads_other_users_memory_and_contaminates():
    store = DictStore()
    policy = PolicyConfig(use_semantic=True, scramble_namespaces=True)
    agent = build_agent(policy, store, StubModel())
    # b states a fact; writes go to b's own namespace.
    agent.run_session("b", _session("b", 0, texts=["my plan_tier is team"]))
    # a is served b's memory (harness picks the neighbor); a's true value is pro.
    p = _probe(expected="pro", avoid=["team"])
    (resp,) = agent.run_session("a", _session("a", 1, probes=[p]), memory_user_id="b")
    r = score_probe(resp["response"], p)
    assert r.contaminated and not r.correct


def test_write_gate_tau_tradeoff_update_tracked_at_07_stale_at_09():
    """The ADR-0004 curve in miniature: tau=0.7 tracks a revision, tau=0.9 refuses it."""
    histories = {}
    for tau in (0.7, 0.9):
        store = DictStore()
        agent = build_agent(PolicyConfig(use_semantic=True, write_gate_tau=tau), store, StubModel())
        agent.run_session("a", _session("a", 0, texts=["my plan_tier is free"]))
        agent.run_session("a", _session("a", 1, texts=["actually my plan_tier is now pro"]))
        p = _probe(expected="pro", avoid=["free"], session=2)
        (resp,) = agent.run_session("a", _session("a", 2, probes=[p]))
        histories[tau] = score_probe(resp["response"], p)
    assert histories[0.7].correct and not histories[0.7].contaminated
    assert not histories[0.9].correct and histories[0.9].contaminated  # stale 'free' surfaced


def test_write_gate_tau_09_rejects_off_vocab_noise():
    store = DictStore()
    agent = build_agent(PolicyConfig(use_semantic=True, write_gate_tau=0.9), store, StubModel())
    agent.run_session("a", _session("a", 0, texts=["my plan_tier is pro",
                                                   "my noise_plan_tier is enterprise"]))
    keys = {r.key for r in store.all("a")}
    assert keys == {"plan_tier"}  # the 0.75-confidence noise fact was refused


def test_episodic_memory_recalls_via_session_summaries():
    store = DictStore()
    agent = build_agent(PolicyConfig(use_episodic=True), store, StubModel())
    agent.run_session("a", _session("a", 0, texts=["my plan_tier is pro", "my timezone is eastern"]))
    p = _probe()
    (resp,) = agent.run_session("a", _session("a", 1, probes=[p]))
    assert score_probe(resp["response"], p).correct
    # And the store holds an episode record, not per-fact semantic records.
    assert all(r.key.startswith("episode:") for r in store.all("a"))


def test_episodic_top_k_recency_loses_old_sessions():
    """Episodic recall is recency-capped: a fact mentioned only in session 0 falls out of the
    top-k episode window — the structural weakness the episodic_only column measures."""
    store = DictStore()
    agent = build_agent(PolicyConfig(use_episodic=True, retrieve_k=3), store, StubModel())
    agent.run_session("a", _session("a", 0, texts=["my plan_tier is pro"]))
    for i in range(1, 6):
        agent.run_session("a", _session("a", i, texts=[f"my noise_field_{i} is thing{i}"]))
    p = _probe(session=6)
    (resp,) = agent.run_session("a", _session("a", 6, probes=[p]))
    assert not score_probe(resp["response"], p).correct


def test_memory_only_context_cannot_answer_same_session_update():
    """ADR-0010: respond sees memory, not the live session — a fact updated in the probe
    session is not yet written when the probe is answered (extract/write run after respond)."""
    store = DictStore()
    agent = build_agent(PolicyConfig(use_semantic=True), store, StubModel())
    agent.run_session("a", _session("a", 0, texts=["my plan_tier is free"]))
    p = _probe(expected="pro", session=1)
    (resp,) = agent.run_session(
        "a", _session("a", 1, texts=["actually my plan_tier is now pro"], probes=[p])
    )
    assert not score_probe(resp["response"], p).correct  # memory still says 'free'
    # ...but the update IS written for future sessions.
    p2 = _probe(expected="pro", session=2)
    (resp2,) = agent.run_session("a", _session("a", 2, probes=[p2]))
    assert score_probe(resp2["response"], p2).correct


def test_model_free_configs_make_no_write_model_calls():
    """memory_off/oracle skip extraction and summarization — cost accounting depends on it."""
    for policy in (PolicyConfig(memory_off=True), PolicyConfig(oracle=True)):
        model = StubModel()
        agent = build_agent(policy, DictStore(), model)
        agent.run_session("a", _session("a", 0, texts=["my plan_tier is pro"]))
        assert model.usage.input_tokens == 0  # no probes, no extraction -> no model calls
