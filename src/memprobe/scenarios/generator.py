"""Program-driven scenario generator — the anti-circularity core (NOTES.md ADR-0005).

The generator, not a model, decides every fact and every turn where a fact is introduced,
updated, or contradicted. That is what makes ground truth trustworthy. An LLM may later
*rephrase* the templated turns for naturalness (realize.py), but it never invents ground
truth. Same seed -> byte-identical structure (test_scenarios.py asserts this).

Implemented deterministically with templates. `realize.py` is the optional phrasing layer.
"""

from __future__ import annotations

import random

from memprobe.config import ScenarioParams
from memprobe.scenarios.schema import (
    FactStatus,
    GroundTruthFact,
    Probe,
    Scenario,
    Session,
    Turn,
)

# A small, transparent fact vocabulary. Each key has a fixed candidate value set so
# metrics.chance_rate can compute an exact guessing floor. Extend, but keep sets closed.
FACT_VOCAB: dict[str, list[str]] = {
    "preferred_contact_channel": ["email", "phone", "sms", "in_app"],
    "plan_tier": ["free", "pro", "team", "enterprise"],
    "timezone": ["eastern", "central", "mountain", "pacific"],
    "primary_device": ["ios", "android", "web", "desktop"],
    "billing_cycle": ["monthly", "annual"],
}


def _distractor_facts(rng: random.Random, user_id: str, n: int, start_id: int) -> list[GroundTruthFact]:
    keys = list(FACT_VOCAB)
    out = []
    for i in range(n):
        key = rng.choice(keys)
        out.append(
            GroundTruthFact(
                fact_id=f"{user_id}-d{start_id + i}",
                user_id=user_id,
                key=f"noise_{key}",  # namespaced so it never collides with a probed key
                value=rng.choice(FACT_VOCAB[key]),
                status=FactStatus.DISTRACTOR,
            )
        )
    return out


def generate_scenario(user_id: str, seed: int, params: ScenarioParams) -> Scenario:
    """Build one user's multi-session scenario deterministically from (user_id, seed).

    For each probed key: introduce a value in an early session; with probability
    `contradiction_rate`, introduce a NEW value in a later session (the old one becomes
    SUPERSEDED). Probes in the final session ask for the current value; superseded values are
    recorded so metrics can detect a stale answer.
    """
    rng = random.Random(f"{user_id}:{seed}")
    keys = list(FACT_VOCAB)
    rng.shuffle(keys)
    probed_keys = keys[: max(2, len(keys) // 2)]

    facts: list[GroundTruthFact] = []
    sessions = [Session(user_id=user_id, index=i) for i in range(params.n_sessions)]

    for k_i, key in enumerate(probed_keys):
        values = FACT_VOCAB[key][:]
        rng.shuffle(values)
        first_val = values[0]
        intro_session = rng.randint(0, max(0, params.n_sessions - 2))
        fid = f"{user_id}-f{k_i}"
        contradicted = rng.random() < params.contradiction_rate

        if contradicted:
            new_val = values[1]
            update_session = rng.randint(intro_session + 1, params.n_sessions - 1)
            facts.append(GroundTruthFact(
                fact_id=fid, user_id=user_id, key=key, value=first_val,
                status=FactStatus.SUPERSEDED, introduced_session=intro_session,
                superseded_by=f"{fid}b",
            ))
            facts.append(GroundTruthFact(
                fact_id=f"{fid}b", user_id=user_id, key=key, value=new_val,
                status=FactStatus.CURRENT, introduced_session=update_session,
            ))
            sessions[intro_session].turns.append(
                Turn(speaker="user", text=f"my {key} is {first_val}", injects=[fid]))
            sessions[update_session].turns.append(
                Turn(speaker="user", text=f"actually my {key} is now {new_val}", injects=[f"{fid}b"]))
            current_val, stale_vals = new_val, [first_val]
        else:
            facts.append(GroundTruthFact(
                fact_id=fid, user_id=user_id, key=key, value=first_val,
                status=FactStatus.CURRENT, introduced_session=intro_session,
            ))
            sessions[intro_session].turns.append(
                Turn(speaker="user", text=f"my {key} is {first_val}", injects=[fid]))
            current_val, stale_vals = first_val, []

        # Probe in the final session: the agent must answer with the CURRENT value and must
        # not surface a superseded one.
        sessions[-1].probes.append(Probe(
            probe_id=f"{fid}-p", session_index=params.n_sessions - 1, fact_key=key,
            expected_value=current_val, must_not_contain=stale_vals,
        ))

    # Distractors: irrelevant facts sprinkled across sessions (retrieval noise).
    n_distractors = int(params.distractor_density * params.n_sessions)
    dfacts = _distractor_facts(rng, user_id, n_distractors, start_id=0)
    facts.extend(dfacts)
    for d in dfacts:
        s = rng.randrange(params.n_sessions)
        sessions[s].turns.append(Turn(speaker="user", text=f"my {d.key} is {d.value}", injects=[d.fact_id]))

    # Pad sessions to the configured turn count with neutral chatter (no injects).
    for s in sessions:
        while len(s.turns) < params.turns_per_session:
            s.turns.append(Turn(speaker="user", text="thanks for the help"))

    return Scenario(user_id=user_id, seed=seed, sessions=sessions, facts=facts)


def generate_suite(seed: int, params: ScenarioParams) -> list[Scenario]:
    """All users for one seed. Cross-user contamination probes are wired by the harness, which
    knows every user's facts (a value belonging to user B becomes B-specific must_not_contain
    when probing user A)."""
    return [generate_scenario(f"u{i:03d}", seed, params) for i in range(params.n_users)]
