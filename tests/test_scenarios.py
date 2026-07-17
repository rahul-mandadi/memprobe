"""Scenario generator: determinism + ground-truth integrity (the anti-circularity core)."""

from memprobe.config import ScenarioParams
from memprobe.scenarios.generator import generate_scenario, generate_suite
from memprobe.scenarios.schema import FactStatus

PARAMS = ScenarioParams(n_users=5, n_sessions=6, turns_per_session=5,
                        contradiction_rate=0.5, distractor_density=0.3, seeds=[0])


def test_same_seed_is_byte_identical():
    a = generate_scenario("u001", 0, PARAMS)
    b = generate_scenario("u001", 0, PARAMS)
    assert a.model_dump() == b.model_dump()


def test_different_seed_differs():
    a = generate_scenario("u001", 0, PARAMS)
    b = generate_scenario("u001", 1, PARAMS)
    assert a.model_dump() != b.model_dump()


def test_exactly_one_current_value_per_probed_key():
    s = generate_scenario("u001", 0, PARAMS)
    for probe in s.sessions[-1].probes:
        currents = [f for f in s.facts if f.key == probe.fact_key and f.status == FactStatus.CURRENT]
        assert len(currents) == 1, f"{probe.fact_key} must have exactly one current value"
        assert probe.expected_value == currents[0].value


def test_superseded_value_is_recorded_as_stale_on_probe():
    # With contradiction_rate=1.0 every probed key must carry a stale value to avoid.
    params = ScenarioParams(n_users=1, n_sessions=6, turns_per_session=5,
                            contradiction_rate=1.0, distractor_density=0.0, seeds=[0])
    s = generate_scenario("u001", 0, params)
    for probe in s.sessions[-1].probes:
        assert probe.must_not_contain, "a contradicted key must list its superseded value"
        assert probe.expected_value not in probe.must_not_contain


def test_current_value_helper_matches_probe():
    s = generate_scenario("u001", 0, PARAMS)
    for probe in s.sessions[-1].probes:
        assert s.current_value(probe.fact_key) == probe.expected_value


def test_suite_has_distinct_users():
    suite = generate_suite(0, PARAMS)
    assert len({s.user_id for s in suite}) == PARAMS.n_users
