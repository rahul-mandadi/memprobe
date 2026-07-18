"""Surface realization: phrasing may change, ground truth and recoverability may not."""

from memprobe.agent.models import StubModel
from memprobe.config import ScenarioParams
from memprobe.eval.metrics import _contains_value
from memprobe.scenarios.generator import generate_scenario
from memprobe.scenarios.realize import realize

PARAMS = ScenarioParams(n_users=1, n_sessions=5, turns_per_session=4,
                        contradiction_rate=0.5, distractor_density=0.3, seeds=[0])


def test_realize_preserves_ground_truth_and_injects():
    s = generate_scenario("u001", 0, PARAMS)
    r = realize(s, StubModel())
    assert r.facts == s.facts  # ground truth is the generator's, never the model's
    for orig_sess, real_sess in zip(s.sessions, r.sessions):
        assert [t.injects for t in orig_sess.turns] == [t.injects for t in real_sess.turns]
        assert orig_sess.probes == real_sess.probes


def test_realize_keeps_values_recoverable_by_the_scorer():
    s = generate_scenario("u001", 0, PARAMS)
    r = realize(s, StubModel())
    facts = {f.fact_id: f for f in r.facts}
    for sess in r.sessions:
        for turn in sess.turns:
            for fid in turn.injects:
                assert _contains_value(turn.text, facts[fid].value), (
                    f"value {facts[fid].value!r} lost from realized turn {turn.text!r}"
                )


def test_realize_falls_back_to_template_when_model_mangles():
    class ManglingModel:
        def complete(self, prompt):
            return "sure, noted!"  # drops every key and value

    s = generate_scenario("u001", 0, PARAMS)
    r = realize(s, ManglingModel())
    # Every inject-bearing turn must have kept its templated text (fail-closed).
    for orig_sess, real_sess in zip(s.sessions, r.sessions):
        for orig, real in zip(orig_sess.turns, real_sess.turns):
            if orig.injects:
                assert real.text == orig.text


def test_realize_does_not_mutate_the_input_scenario():
    s = generate_scenario("u001", 0, PARAMS)
    before = s.model_dump()
    realize(s, StubModel())
    assert s.model_dump() == before
