"""LLM-judge (secondary metric) + the mandatory agreement audit (ADR-0005)."""

from memprobe.agent.models import StubModel
from memprobe.eval.judge import AgreementAudit, audit_agreement, judge_quality


class FakeJudge:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt):
        return self.reply


def test_parses_model_rating_on_ten_scale():
    score = judge_quality("you are on pro", "pro", FakeJudge("rating: 8\nclear and direct."))
    assert abs(score.quality - 0.8) < 1e-9
    assert "clear and direct" in score.rationale
    assert not score.rationale.startswith("[fallback")


def test_parses_unit_scale_and_clamps():
    assert judge_quality("r", "e", FakeJudge("rating: 0.9")).quality == 0.9
    assert judge_quality("r", "e", FakeJudge("rating: 37")).quality == 1.0  # clamped


def test_unparseable_reply_falls_back_to_flagged_slot_check():
    good = judge_quality("plan_tier: pro", "pro", StubModel())
    bad = judge_quality("i don't have that on file", "pro", StubModel())
    assert good.quality == 1.0 and bad.quality == 0.0
    assert good.rationale.startswith("[fallback slot-check]")
    assert bad.rationale.startswith("[fallback slot-check]")


# --- agreement audit ---------------------------------------------------------------------

def test_perfect_agreement_is_kappa_one():
    a = audit_agreement({"x": 1.0, "y": 0.0, "z": 1.0}, {"x": 0.9, "y": 0.1, "z": 0.8})
    assert a.n_sampled == 3 and a.agreement == 1.0 and a.cohen_kappa == 1.0
    assert a.trustworthy()


def test_chance_level_agreement_is_kappa_zero():
    # Balanced raters agreeing exactly half the time: po=0.5, pe=0.5 -> kappa 0.
    h = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0}
    j = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}
    a = audit_agreement(h, j)
    assert a.agreement == 0.5 and abs(a.cohen_kappa) < 1e-9
    assert not a.trustworthy()


def test_constant_opposite_raters_get_zero_kappa():
    a = audit_agreement({"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 0.0})
    assert a.agreement == 0.0 and a.cohen_kappa == 0.0


def test_audit_joins_on_shared_keys_only():
    a = audit_agreement({"x": 1.0, "only_human": 1.0}, {"x": 1.0, "only_judge": 0.0})
    assert a.n_sampled == 1


def test_empty_intersection_is_degenerate_and_untrustworthy():
    a = audit_agreement({}, {})
    assert a == AgreementAudit(n_sampled=0, agreement=0.0, cohen_kappa=0.0)
    assert not a.trustworthy()
