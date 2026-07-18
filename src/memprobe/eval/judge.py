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


_JUDGE_INSTRUCTION = (
    "You are auditing a customer-support reply for quality (clarity, directness, and whether "
    "it communicates the expected account detail). Respond with ONLY a line "
    "'rating: <number 0-10>' followed by one short sentence of rationale."
)


def _parse_rating(raw: str) -> float | None:
    """First number in the reply, mapped to [0, 1]; None if the model gave no number."""
    import re

    m = re.search(r"(\d+(?:\.\d+)?)", raw)
    if not m:
        return None
    val = float(m.group(1))
    if val > 1.0:  # models usually answer on the asked 0-10 scale
        val /= 10.0
    return max(0.0, min(1.0, val))


def judge_quality(response: str, probe_expected: str, model) -> JudgeScore:
    """Rate response quality with the judge model (milestone 4).

    If the model's reply carries no parseable rating (the deterministic stub never rates),
    fall back to a transparent slot-check so the secondary pipeline stays runnable — and
    tag the rationale so a fallback can NEVER masquerade as a model judgment downstream.
    """
    raw = model.complete(
        f"{_JUDGE_INSTRUCTION}\n\nExpected account detail: {probe_expected}\nReply: {response}"
    )
    quality = _parse_rating(raw)
    if quality is None:
        from memprobe.eval.metrics import _contains_value

        quality = 1.0 if _contains_value(response, probe_expected) else 0.0
        return JudgeScore(quality=quality, rationale=f"[fallback slot-check] {raw.strip()[:160]}")
    return JudgeScore(quality=quality, rationale=raw.strip()[:240])


def audit_agreement(
    human_labels: dict[str, float], judge_labels: dict[str, float], threshold: float = 0.5
) -> AgreementAudit:
    """Compare judge to human labels on the same items (milestone 4).

    Labels are binarized at `threshold` (quality >= 0.5 reads as "acceptable") because raw
    percent-agreement on floats is ill-defined and kappa needs categories. Cohen's kappa is
    computed by hand — it is four lines, and depending on sklearn for it would be the
    heaviest import in the repo. Degenerate case: perfect observed agreement is kappa 1.0
    even when chance agreement is also 1.0 (the standard 0/0 convention); two constant,
    opposite raters get pe=0 so kappa=po=0.

    Must be run and its result printed alongside any judge-derived number in the report
    (ADR-0005: an unaudited judge is a vibe, not a metric).
    """
    keys = sorted(set(human_labels) & set(judge_labels))
    n = len(keys)
    if n == 0:
        return AgreementAudit(n_sampled=0, agreement=0.0, cohen_kappa=0.0)
    h = [human_labels[k] >= threshold for k in keys]
    j = [judge_labels[k] >= threshold for k in keys]
    po = sum(hi == ji for hi, ji in zip(h, j)) / n
    p_h, p_j = sum(h) / n, sum(j) / n
    pe = p_h * p_j + (1 - p_h) * (1 - p_j)
    if po >= 1.0:
        kappa = 1.0
    elif pe >= 1.0:
        kappa = 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    return AgreementAudit(n_sampled=n, agreement=po, cohen_kappa=kappa)
