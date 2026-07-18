"""Semantic memory: extract user facts from turns and write them under a confidence gate.

The write-gate is the star ablation axis (NOTES.md ADR-0004): a candidate fact is written only
if extraction confidence >= tau. Sweeping tau (0.7 vs 0.9 in the default matrix) turns a vague
"we gate risky writes" feature into a measured curve: high tau -> fewer false memories
(less contamination) but more misses (lower task success). That trade-off IS the result.

`apply_write_gate` is implemented (pure policy). Extraction needs a model and is stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass

from memprobe.memory.store import MemoryRecord, MemoryStore


@dataclass
class CandidateFact:
    key: str
    value: str
    confidence: float
    source_session: int


def apply_write_gate(
    store: MemoryStore,
    user_id: str,
    candidates: list[CandidateFact],
    tau: float,
    now: float,
) -> int:
    """Write only candidates with confidence >= tau. Returns count written. Implemented.

    This is the whole gate: transparent, one comparison. A later value for the same key
    supersedes earlier ones at read time via provenance/timestamp (retrieval + forgetting
    handle recency), so the gate's only job is the accept/reject decision.
    """
    written = 0
    for c in candidates:
        if c.confidence >= tau:
            store.put(MemoryRecord(
                key=c.key, value=c.value, user_id=user_id,
                source_session=c.source_session, written_at=now, confidence=c.confidence,
            ))
            written += 1
    return written


# --- Evidence-grounded confidence tiers (ADR-0008) -------------------------------------
# Confidence is computed by the EXTRACTOR from source-text evidence, never self-reported by
# the model. Rationale: (a) it is a real, varying signal on the deterministic stub path, so
# the tau ablation measures something; (b) holding the confidence definition constant across
# model backends means a tau sweep ablates the GATE policy, not one model's calibration
# quirks; (c) a hallucinated extraction has no supporting assertion in the source turns and
# dies at the gate — the gate is auditable evidence, not vibes.
CONF_VOCAB_ASSERT = 0.95  # clean direct assertion; known key; value inside its closed set
CONF_UPDATE = 0.85        # correction/update assertion ("is now" / "actually"): revisions
                          # of a stored belief flip again more often, so a calibrated
                          # extractor discounts them — this is what makes tau=0.9 measurably
                          # conservative (rejects revisions) vs tau=0.7 (tracks them)
CONF_OFF_VOCAB = 0.75     # clean assertion of a key OUTSIDE the closed vocabulary — cannot
                          # be schema-validated, so it is exactly the "plausible junk" a
                          # strict gate should refuse (distractors land here)
CONF_WEAK = 0.55          # hedged phrasing, or a known key with an out-of-set value
CONF_NO_EVIDENCE = 0.30   # the model proposed it but no source turn asserts it
CORROBORATION_BONUS = 0.02  # per additional independent assertion of the same (key, value)
CONF_CAP = 0.98

_HEDGES = ("i think", "maybe", "probably", "not sure", "i guess", "if i remember")

_EXTRACT_INSTRUCTION = (
    "List every stable user fact stated in this conversation, one per line, formatted "
    "exactly '<key>: <value>' with the key in snake_case as the user named it. "
    "Only include facts the user actually stated.\n\nConversation:\n"
)


def _assertion_evidence(key: str, value: str, texts: list[str]) -> tuple[int, bool, bool]:
    """(n_assertions, any_update_form, any_hedged) for '<key> is <value>' in source turns."""
    import re

    n, update_form, hedged = 0, False, False
    pat = re.compile(rf"\bmy {re.escape(key)} is (now )?{re.escape(value)}\b")
    for t in texts:
        low = t.lower()
        m = pat.search(low)
        if not m:
            continue
        n += 1
        if m.group(1) or "actually" in low:
            update_form = True
        if any(h in low for h in _HEDGES):
            hedged = True
    return n, update_form, hedged


def extract_facts(turns, model, *, session_index: int = 0, vocab=None) -> list[CandidateFact]:
    """Extract candidate (key, value, confidence) facts from conversation turns via the model.

    The MODEL proposes candidates (it reads the conversation and lists 'key: value' facts —
    the deterministic StubModel does this by construction, a real model by instruction). The
    EXTRACTOR then assigns confidence from source-text evidence using the tier table above.
    Confidence is a real signal, not a constant — the tau ablation depends on it (ADR-0004).

    `vocab` is the domain's closed value sets (defaults to the generator's FACT_VOCAB);
    injectable so a custom domain can bring its own schema.
    """
    import re

    user_texts = [t.text for t in turns if getattr(t, "speaker", "user") == "user"]
    if not user_texts:
        return []
    if vocab is None:
        from memprobe.scenarios.generator import FACT_VOCAB as vocab

    raw = model.complete(_EXTRACT_INSTRUCTION + "\n".join(user_texts))

    # Parse '<key>: <value>' pairs (and tolerate prose 'my <key> is <value>' restatements).
    pairs: list[tuple[str, str]] = re.findall(r"\b([\w]+)\s*:\s*([\w]+)\b", raw.lower())
    pairs += re.findall(r"\bmy ([\w]+) is (?:now )?([\w]+)\b", raw.lower())
    seen: set[tuple[str, str]] = set()

    out: list[CandidateFact] = []
    for key, value in pairs:
        if (key, value) in seen:
            continue
        seen.add((key, value))
        n, update_form, hedged = _assertion_evidence(key, value, user_texts)
        if n == 0:
            conf = CONF_NO_EVIDENCE
        else:
            if key in vocab:
                conf = CONF_VOCAB_ASSERT if value in vocab[key] else CONF_WEAK
            else:
                conf = CONF_OFF_VOCAB
            if update_form:
                conf = min(conf, CONF_UPDATE)
            if hedged:
                conf = min(conf, CONF_WEAK)
            conf = min(conf + CORROBORATION_BONUS * (n - 1), CONF_CAP)
        out.append(CandidateFact(key=key, value=value, confidence=conf,
                                 source_session=session_index))
    return out
