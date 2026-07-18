"""Deterministic metrics — the credibility core. Scored on hand-built fixtures."""

from memprobe.eval.metrics import _contains_value, aggregate, score_probe
from memprobe.scenarios.schema import Probe


def _probe(expected, avoid=None):
    return Probe(probe_id="p1", session_index=0, fact_key="plan_tier",
                expected_value=expected, must_not_contain=avoid or [])


def test_correct_answer():
    r = score_probe("your plan tier is pro", _probe("pro"))
    assert r.correct and not r.contaminated and not r.stale


def test_word_boundary_prevents_false_match():
    # "pro" must not match inside "problem"; the agent didn't actually state the value.
    r = score_probe("no problem, let me check", _probe("pro"))
    assert not r.correct


def test_contamination_detected():
    # surfaces another user's value ("enterprise") -> contaminated, and since not the expected
    # current value, also flagged stale-or-wrong via the shared must_not_contain channel.
    r = score_probe("you're on enterprise", _probe("pro", avoid=["enterprise"]))
    assert r.contaminated and not r.correct


def test_correct_despite_listing_avoided_only_counts_contamination_when_present():
    r = score_probe("your plan tier is pro", _probe("pro", avoid=["free"]))
    assert r.correct and not r.contaminated


def test_aggregate_rates():
    results = [
        score_probe("pro", _probe("pro")),
        score_probe("free", _probe("pro", avoid=["free"])),  # wrong + contaminated
        score_probe("pro", _probe("pro")),
    ]
    m = aggregate(results)
    assert m.n_probes == 3
    assert abs(m.task_success - 2 / 3) < 1e-9
    assert abs(m.contamination_rate - 1 / 3) < 1e-9


def test_contains_value_normalizes_punctuation_and_case():
    assert _contains_value("Plan: PRO.", "pro")
    assert not _contains_value("", "pro")


# --- chance_rate: the exact calibration floor (milestone 1) ---

def _mini_scenario(facts=None):
    from memprobe.scenarios.schema import Scenario
    return Scenario(user_id="u", seed=0, sessions=[], facts=facts or [])


def test_chance_rate_empty_probes_is_zero():
    from memprobe.eval.metrics import chance_rate
    assert chance_rate(_mini_scenario(), []) == 0.0


def test_chance_rate_binary_key_is_half():
    # billing_cycle has exactly 2 candidate values -> guessing floor 0.5.
    from memprobe.eval.metrics import chance_rate
    p = Probe(probe_id="p", session_index=0, fact_key="billing_cycle", expected_value="annual")
    assert abs(chance_rate(_mini_scenario(), [p]) - 0.5) < 1e-12


def test_chance_rate_mixes_vocab_sizes():
    # plan_tier (4 values) + billing_cycle (2 values) -> mean(0.25, 0.5) = 0.375.
    from memprobe.eval.metrics import chance_rate
    ps = [
        Probe(probe_id="a", session_index=0, fact_key="plan_tier", expected_value="pro"),
        Probe(probe_id="b", session_index=0, fact_key="billing_cycle", expected_value="annual"),
    ]
    assert abs(chance_rate(_mini_scenario(), ps) - 0.375) < 1e-12


def test_chance_rate_unknown_key_falls_back_to_observed_values():
    # A custom-domain key outside FACT_VOCAB uses the scenario's own observed value set.
    from memprobe.eval.metrics import chance_rate
    from memprobe.scenarios.schema import GroundTruthFact
    facts = [
        GroundTruthFact(fact_id="f1", user_id="u", key="favorite_shade", value="teal"),
        GroundTruthFact(fact_id="f2", user_id="u", key="favorite_shade", value="mauve",
                        status="superseded"),
    ]
    p = Probe(probe_id="p", session_index=0, fact_key="favorite_shade", expected_value="teal")
    assert abs(chance_rate(_mini_scenario(facts), [p]) - 0.5) < 1e-12


def test_chance_rate_unseen_key_contributes_zero():
    from memprobe.eval.metrics import chance_rate
    p = Probe(probe_id="p", session_index=0, fact_key="never_mentioned", expected_value="x")
    assert chance_rate(_mini_scenario(), [p]) == 0.0
