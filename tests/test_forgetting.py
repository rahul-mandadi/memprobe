"""Forgetting policy (milestone 4): floor-drop + freshest-first ranking, wired in retrieval."""

import pytest

from memprobe.memory.forgetting import FORGET_FLOOR, apply_forgetting, recency_weight
from memprobe.memory.store import DictStore, MemoryRecord


def _rec(key, value, at):
    return MemoryRecord(key=key, value=value, user_id="a", source_session=int(at // 10), written_at=at)


def test_none_policy_is_identity():
    records = [_rec("plan_tier", "free", 0.0), _rec("plan_tier", "pro", 50.0)]
    assert apply_forgetting(records, now=1000.0, policy="none") == records
    assert apply_forgetting(records, now=1000.0, policy=None) == records


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        apply_forgetting([], now=0.0, policy="anterograde")


def test_recency_decay_drops_below_floor_and_ranks_freshest_first():
    old = _rec("plan_tier", "free", 0.0)     # 105 days old at now=105 -> weight ~0.088
    mid = _rec("timezone", "eastern", 60.0)  # 45 days -> ~0.354
    new = _rec("plan_tier", "pro", 100.0)    # 5 days -> ~0.891
    out = apply_forgetting([old, new, mid], now=105.0, policy="recency_decay", half_life=30.0)
    assert out == [new, mid]  # old dropped, survivors freshest-first
    assert recency_weight(old, 105.0, 30.0) < FORGET_FLOOR <= recency_weight(mid, 105.0, 30.0)


def test_floor_is_a_sharp_threshold():
    # weight(69d, hl=30) ~= 0.203 > 0.2 kept; weight(71d) ~= 0.194 < 0.2 dropped.
    rec = _rec("plan_tier", "pro", 0.0)
    assert apply_forgetting([rec], now=69.0, policy="recency_decay", half_life=30.0) == [rec]
    assert apply_forgetting([rec], now=71.0, policy="recency_decay", half_life=30.0) == []


def test_disabled_half_life_never_drops():
    old = _rec("plan_tier", "free", 0.0)
    assert apply_forgetting([old], now=1e6, policy="recency_decay", half_life=0.0) == [old]


def test_graph_forgetting_loses_old_facts_and_keeps_fresh_ones():
    """The ablation's cost side: an old-but-still-current fact fades out of retrieval."""
    from memprobe.agent.graph import PolicyConfig, build_agent
    from memprobe.agent.models import StubModel
    from memprobe.eval.metrics import score_probe
    from memprobe.scenarios.schema import Probe, Session, Turn

    def _session(i, texts=(), probes=()):
        return Session(user_id="a", index=i,
                       turns=[Turn(speaker="user", text=t) for t in texts], probes=list(probes))

    store = DictStore()
    policy = PolicyConfig(use_semantic=True, forgetting="recency_decay", half_life_days=30.0)
    agent = build_agent(policy, store, StubModel())
    # 10-day session spacing (the harness clock): session 0 at day 0, session 9 at day 90.
    agent.run_session("a", _session(0, texts=["my plan_tier is pro"]), now=0.0)
    agent.run_session("a", _session(8, texts=["my timezone is eastern"]), now=80.0)

    old_probe = Probe(probe_id="o", session_index=9, fact_key="plan_tier", expected_value="pro")
    fresh_probe = Probe(probe_id="f", session_index=9, fact_key="timezone", expected_value="eastern")
    responses = agent.run_session("a", _session(9, probes=[old_probe, fresh_probe]), now=90.0)
    by_id = {r["probe_id"]: r["response"] for r in responses}
    assert not score_probe(by_id["o"], old_probe).correct   # 90 days old -> forgotten
    assert score_probe(by_id["f"], fresh_probe).correct     # 10 days old -> retained

    # Control: identical history without forgetting answers both.
    store2 = DictStore()
    agent2 = build_agent(PolicyConfig(use_semantic=True), store2, StubModel())
    agent2.run_session("a", _session(0, texts=["my plan_tier is pro"]), now=0.0)
    agent2.run_session("a", _session(8, texts=["my timezone is eastern"]), now=80.0)
    responses2 = agent2.run_session("a", _session(9, probes=[old_probe, fresh_probe]), now=90.0)
    assert all(
        score_probe(r["response"], p).correct
        for r, p in zip(sorted(responses2, key=lambda r: r["probe_id"]),
                        sorted([old_probe, fresh_probe], key=lambda p: p.probe_id))
    )
