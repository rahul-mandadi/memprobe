"""Store isolation, write-gate policy, forgetting decay, and config->policy mapping."""

from memprobe.agent.policies import resolve
from memprobe.config import Anchor, MemConfig
from memprobe.memory.forgetting import recency_weight
from memprobe.memory.semantic import CandidateFact, apply_write_gate
from memprobe.memory.store import DictStore, MemoryRecord


def test_cross_user_isolation():
    store = DictStore()
    store.put(MemoryRecord(key="plan_tier", value="pro", user_id="a", source_session=0, written_at=0))
    store.put(MemoryRecord(key="plan_tier", value="free", user_id="b", source_session=0, written_at=0))
    assert [r.value for r in store.all("a")] == ["pro"]
    assert [r.value for r in store.all("b")] == ["free"]
    assert store.get("a", "plan_tier")[0].value == "pro"


def test_write_gate_respects_tau():
    store = DictStore()
    cands = [
        CandidateFact(key="plan_tier", value="pro", confidence=0.95, source_session=0),
        CandidateFact(key="timezone", value="pacific", confidence=0.60, source_session=0),
    ]
    written = apply_write_gate(store, "a", cands, tau=0.7, now=1.0)
    assert written == 1
    assert {r.key for r in store.all("a")} == {"plan_tier"}


def test_write_gate_high_tau_writes_nothing():
    store = DictStore()
    cands = [CandidateFact(key="plan_tier", value="pro", confidence=0.8, source_session=0)]
    assert apply_write_gate(store, "a", cands, tau=0.9, now=1.0) == 0


def test_recency_weight_halves_at_half_life():
    rec = MemoryRecord(key="k", value="v", user_id="a", source_session=0, written_at=0.0)
    assert abs(recency_weight(rec, now=30.0, half_life=30.0) - 0.5) < 1e-9
    assert recency_weight(rec, now=0.0, half_life=30.0) == 1.0
    assert recency_weight(rec, now=100.0, half_life=0.0) == 1.0  # disabled


def test_resolve_anchor_flags():
    off = resolve(MemConfig(name="memory_off", anchor=Anchor.CALIBRATION, memory="none"))
    assert off.memory_off and not off.oracle
    orc = resolve(MemConfig(name="oracle", anchor=Anchor.UPPER_BOUND, memory="oracle"))
    assert orc.oracle and not orc.memory_off
    plc = resolve(MemConfig(name="placebo", anchor=Anchor.PLACEBO, memory="semantic",
                            scramble_namespaces=True))
    assert plc.scramble_namespaces


def test_resolve_semantic_gate():
    cfg = MemConfig(name="s", memory=["episodic", "semantic"], retrieval="embedding",
                    write_gate_tau=0.9)
    p = resolve(cfg)
    assert p.use_episodic and p.use_semantic
    assert p.retrieval == "embedding" and p.write_gate_tau == 0.9
