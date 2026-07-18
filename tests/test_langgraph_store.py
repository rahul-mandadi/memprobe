"""Backend contract + parity: DictStore and LangGraphStore must behave identically.

Parity is not cosmetic — the direct-vs-embedding ablation is only apples-to-apples if the
store surface ranks the same way regardless of substrate (store.py docstring).
"""

import pytest

from memprobe.memory.store import DictStore, LangGraphStore, MemoryRecord


def _fill(store):
    store.put(MemoryRecord(key="plan_tier", value="free", user_id="a", source_session=0, written_at=0.0))
    store.put(MemoryRecord(key="noise_plan_tier", value="team", user_id="a", source_session=1, written_at=1.0))
    store.put(MemoryRecord(key="plan_tier", value="pro", user_id="a", source_session=3, written_at=3.0))
    store.put(MemoryRecord(key="plan_tier", value="enterprise", user_id="b", source_session=0, written_at=0.0))
    return store


@pytest.mark.parametrize("backend", [DictStore, LangGraphStore])
class TestStoreContract:
    def test_append_semantics_keep_history(self, backend):
        s = _fill(backend())
        # Two plan_tier records for user a — a superseded value must stay observable.
        assert [r.value for r in s.get("a", "plan_tier")] == ["free", "pro"]

    def test_cross_user_isolation(self, backend):
        s = _fill(backend())
        assert {r.value for r in s.all("a")} == {"free", "team", "pro"}
        assert {r.value for r in s.all("b")} == {"enterprise"}
        assert s.all("nobody") == []

    def test_search_substring_match_most_recent_first(self, backend):
        s = _fill(backend())
        # 'noise_plan_tier' contains 'plan_tier' -> it is retrieval noise by design.
        assert [r.value for r in s.search("a", "plan_tier")] == ["pro", "team", "free"]
        assert [r.value for r in s.search("a", "plan_tier", k=2)] == ["pro", "team"]
        assert s.search("a", "billing_cycle") == []

    def test_round_trip_preserves_record_fields(self, backend):
        s = backend()
        rec = MemoryRecord(key="k", value="v", user_id="u", source_session=4,
                           written_at=40.0, confidence=0.85, meta={"type": "semantic"})
        s.put(rec)
        (got,) = s.all("u")
        assert got == rec


def test_backend_parity_field_by_field():
    a, b = _fill(DictStore()), _fill(LangGraphStore())
    for uid in ("a", "b"):
        assert a.all(uid) == b.all(uid)
        assert a.get(uid, "plan_tier") == b.get(uid, "plan_tier")
        assert a.search(uid, "plan_tier", k=3) == b.search(uid, "plan_tier", k=3)


def test_langgraph_store_uses_the_langgraph_substrate():
    # ADR-0003 is 'build ON langgraph.store' — assert the adapter actually does.
    from langgraph.store.memory import InMemoryStore

    s = LangGraphStore()
    assert isinstance(s._store, InMemoryStore)
    s.put(MemoryRecord(key="k", value="v", user_id="u", source_session=0, written_at=0.0))
    assert s._store.search(("memprobe", "u"), limit=10), "records must live in the substrate"
