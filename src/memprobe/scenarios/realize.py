"""Optional LLM surface-realization of templated turns.

The generator emits robotic turns ("my plan_tier is pro"). This layer optionally rephrases
them into natural support-chat language WITHOUT changing which facts a turn encodes — the
`injects` list is the contract and must be preserved. If no model is configured, the templated
text is used as-is, so the whole lab runs model-free (important for hermetic tests and for the
calibration check on templated-vs-realized scenarios, NOTES.md open questions).

Stubbed — needs a model backend.
"""

from __future__ import annotations

from memprobe.scenarios.schema import Scenario


def _recoverable(text: str, key: str, value: str) -> bool:
    """The injects invariant, checked with the SAME matcher scoring uses: if metrics.py can
    still find the value (and the key phrase) in the rephrased text, scoring stays sound.
    Reusing _contains_value is deliberate — 'recoverable' must mean recoverable BY THE
    SCORER, not by some looser notion."""
    from memprobe.eval.metrics import _contains_value

    return _contains_value(text, value) and _contains_value(
        text.replace("_", " "), key.replace("_", " ")
    )


def realize(scenario: Scenario, model) -> Scenario:
    """Return a copy with naturalistic turn text; `injects`/ground truth unchanged.

    Fail-closed per turn: the model's rephrase is accepted ONLY if every injected fact's key
    and value survive verbatim (checked with the scorer's own matcher); otherwise the
    templated text stays. The generator's ground truth is never touched — an LLM here can
    change phrasing, never facts (ADR-0005). With no model (or a rejected rephrase) the
    templated fallback is the output, so the whole lab keeps running model-free.
    """
    out = scenario.model_copy(deep=True)
    facts_by_id = {f.fact_id: f for f in out.facts}
    for session in out.sessions:
        for turn in session.turns:
            if turn.speaker != "user" or not turn.injects:
                continue  # neutral chatter needs no realization; agent turns carry no facts
            injected = [facts_by_id[fid] for fid in turn.injects if fid in facts_by_id]
            if not injected:
                continue
            must_keep = ", ".join(f"'{f.key}' and '{f.value}'" for f in injected)
            rephrased = model.complete(
                "Rephrase this customer-support message so it sounds natural and "
                f"conversational. You MUST keep these exact terms verbatim: {must_keep}. "
                f"Reply with only the rephrased message.\n\nMessage: {turn.text}"
            ).strip()
            if rephrased and all(_recoverable(rephrased, f.key, f.value) for f in injected):
                turn.text = rephrased
            # else: keep the templated text — the invariant outranks naturalness
    return out
