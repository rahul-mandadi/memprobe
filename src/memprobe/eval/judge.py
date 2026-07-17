"""Secondary LLM-as-judge for response QUALITY, with a mandatory agreement audit.

This never produces a headline metric (those are deterministic — metrics.py). The judge only
rates response quality (helpfulness/coherence) where a slot-check can't. Every judge number
reported anywhere MUST carry the agreement-audit result: %agreement against a hand-labeled
sample. A judge you haven't audited against human labels is a vibe, not a metric
(NOTES.md ADR-0005).

Stubbed — needs a model backend (agent.models).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JudgeScore:
    quality: float          # 0..1 rated response quality
    rationale: str


@dataclass
class AgreementAudit:
    """Result of comparing judge labels to human labels on a sampled subset."""

    n_sampled: int
    agreement: float        # fraction where judge label == human label
    cohen_kappa: float      # chance-corrected agreement

    def trustworthy(self, min_agreement: float = 0.8) -> bool:
        return self.agreement >= min_agreement


def judge_quality(response: str, probe_expected: str, model) -> JudgeScore:
    """Rate response quality with the judge model. TODO(milestone-4)."""
    raise NotImplementedError("TODO(milestone-4): judge prompt + parse; model via agent.models")


def audit_agreement(human_labels: dict[str, float], judge_labels: dict[str, float]) -> AgreementAudit:
    """Compare judge to human labels on the same items. TODO(milestone-4).

    Must be run and its result printed alongside any judge-derived number in the report.
    """
    raise NotImplementedError("TODO(milestone-4): compute agreement + Cohen's kappa")
