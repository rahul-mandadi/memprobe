"""Fact extraction (evidence-grounded confidence, ADR-0008) + episodic summaries."""

from memprobe.agent.models import StubModel
from memprobe.memory import episodic
from memprobe.memory.semantic import (
    CONF_CAP,
    CONF_NO_EVIDENCE,
    CONF_OFF_VOCAB,
    CONF_UPDATE,
    CONF_VOCAB_ASSERT,
    CONF_WEAK,
    CORROBORATION_BONUS,
    extract_facts,
)
from memprobe.memory.store import DictStore
from memprobe.scenarios.schema import Turn


def _turns(*texts):
    return [Turn(speaker="user", text=t) for t in texts]


def test_clean_vocab_assertion_scores_highest():
    (c,) = extract_facts(_turns("my plan_tier is pro"), StubModel())
    assert (c.key, c.value, c.confidence) == ("plan_tier", "pro", CONF_VOCAB_ASSERT)


def test_update_assertion_is_discounted():
    (c,) = extract_facts(_turns("actually my plan_tier is now team"), StubModel())
    assert c.confidence == CONF_UPDATE


def test_off_vocab_key_lands_between_weak_and_update():
    (c,) = extract_facts(_turns("my noise_timezone is central"), StubModel())
    assert c.confidence == CONF_OFF_VOCAB
    assert CONF_WEAK < CONF_OFF_VOCAB < CONF_UPDATE


def test_known_key_with_out_of_set_value_is_weak():
    (c,) = extract_facts(_turns("my plan_tier is platinum"), StubModel())
    assert c.confidence == CONF_WEAK


def test_hedged_assertion_is_weak():
    (c,) = extract_facts(_turns("i think my plan_tier is pro"), StubModel())
    assert c.confidence == CONF_WEAK


def test_corroboration_bumps_confidence():
    turns = _turns("my plan_tier is pro", "thanks", "my plan_tier is pro")
    (c,) = extract_facts(turns, StubModel())
    assert abs(c.confidence - min(CONF_VOCAB_ASSERT + CORROBORATION_BONUS, CONF_CAP)) < 1e-12


def test_hallucinated_candidate_has_no_evidence_and_dies_at_any_gate():
    class HallucinatingModel:
        def complete(self, prompt):
            return "plan_tier: enterprise"  # never asserted in the turns

        usage = None

    (c,) = extract_facts(_turns("my timezone is eastern"), HallucinatingModel())
    assert c.confidence == CONF_NO_EVIDENCE  # below every tau in the matrix


def test_agent_turns_are_not_fact_sources():
    turns = [Turn(speaker="agent", text="my plan_tier is pro")]
    assert extract_facts(turns, StubModel()) == []


def test_session_index_is_stamped():
    (c,) = extract_facts(_turns("my plan_tier is pro"), StubModel(), session_index=5)
    assert c.source_session == 5


# --- episodic ---------------------------------------------------------------------------

def test_summarize_session_recaps_latest_value_per_key():
    turns = _turns("my plan_tier is free", "my plan_tier is now pro", "my timezone is eastern")
    summary = episodic.summarize_session(turns, StubModel())
    assert "plan_tier: pro" in summary and "timezone: eastern" in summary
    assert "free" not in summary  # latest value only


def test_summarize_empty_session_is_empty():
    assert episodic.summarize_session([], StubModel()) == ""


def test_write_episode_persists_a_prefixed_unconditional_record():
    store = DictStore()
    episodic.write_episode(store, "a", 3, "plan_tier: pro", now=30.0)
    (rec,) = store.all("a")
    assert rec.key == "episode:003"
    assert rec.confidence == 1.0 and rec.meta["type"] == "episodic"
    assert rec.written_at == 30.0 and rec.source_session == 3


def test_write_episode_skips_empty_summaries():
    store = DictStore()
    episodic.write_episode(store, "a", 0, "", now=0.0)
    assert store.all("a") == []


def test_episode_keys_never_collide_with_direct_fact_search():
    store = DictStore()
    episodic.write_episode(store, "a", 0, "plan_tier: pro", now=0.0)
    # A direct search for any fact key must not surface episode records.
    assert store.search("a", "plan_tier") == []
