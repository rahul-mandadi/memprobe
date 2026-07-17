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


def realize(scenario: Scenario, model) -> Scenario:
    """Return a copy with naturalistic turn text; `injects`/ground truth unchanged.

    Invariant to preserve and TEST: for every turn, the set of injected fact values must still
    be recoverable from the rephrased text (else scoring breaks). TODO(milestone-2).
    """
    raise NotImplementedError("TODO(milestone-2): LLM rephrase preserving injects invariant")
