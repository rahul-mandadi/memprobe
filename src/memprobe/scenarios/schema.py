"""Typed scenario schema.

This is the ground-truth contract for the whole lab. Everything downstream — the agent, the
metrics, the anchors — reads these objects. Ground truth is carried explicitly on the data
structures (which fact is current, which turn introduced or contradicted it), so scoring is a
deterministic lookup, never a model call (NOTES.md ADR-0005).

Fully implemented: these are plain typed records with no model dependency.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FactStatus(str, Enum):
    """State of a fact at a given point in a scenario timeline."""

    CURRENT = "current"       # the true, up-to-date value the agent should act on
    SUPERSEDED = "superseded"  # was true earlier, later updated/contradicted -> now STALE
    DISTRACTOR = "distractor"  # irrelevant noise; must never drive an answer


class GroundTruthFact(BaseModel):
    """A single piece of user-specific ground truth the generator injected.

    The (key, value) is what the agent must eventually get right for `key`. `status` and
    `introduced_session` let metrics distinguish a correct current answer from a stale one
    and let the generator place contradictions at known points.
    """

    fact_id: str
    user_id: str
    key: str                      # e.g. "preferred_contact_channel"
    value: str                    # e.g. "email"
    status: FactStatus = FactStatus.CURRENT
    introduced_session: int = 0   # session index where this value first appears
    superseded_by: str | None = None  # fact_id that overrides this one, if any


class Turn(BaseModel):
    """One utterance in a session. `injects` links surface text to the fact(s) it encodes."""

    speaker: str                  # "user" | "agent"
    text: str
    injects: list[str] = Field(default_factory=list)  # fact_ids encoded in this turn


class Probe(BaseModel):
    """A question posed to the agent whose correct answer is a known ground-truth value.

    The deterministic metric checks whether the agent's response satisfies `expected_value`
    for `fact_key`. `must_not_contain` carries other-user or superseded values that would
    count as contamination / staleness if they surface.
    """

    probe_id: str
    session_index: int
    fact_key: str
    expected_value: str            # the CURRENT ground-truth value
    must_not_contain: list[str] = Field(default_factory=list)


class Session(BaseModel):
    user_id: str
    index: int
    turns: list[Turn] = Field(default_factory=list)
    probes: list[Probe] = Field(default_factory=list)


class Scenario(BaseModel):
    """One synthetic user's full multi-session history plus the ground truth to score against.

    `seed` and the generator params make a scenario fully reproducible — the same seed must
    regenerate byte-identical structure (test_scenarios.py asserts this).
    """

    user_id: str
    seed: int
    sessions: list[Session] = Field(default_factory=list)
    facts: list[GroundTruthFact] = Field(default_factory=list)

    def current_value(self, key: str) -> str | None:
        """The single CURRENT ground-truth value for a fact key (None if unknown)."""
        for f in self.facts:
            if f.key == key and f.status == FactStatus.CURRENT:
                return f.value
        return None

    def stale_values(self, key: str) -> list[str]:
        """All superseded values for a key — surfacing any of these is a staleness error."""
        return [f.value for f in self.facts if f.key == key and f.status == FactStatus.SUPERSEDED]
