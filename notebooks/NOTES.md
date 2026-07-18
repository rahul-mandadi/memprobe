# Design notes & ADRs

Architecture decision records for memprobe. Each one exists because a stress-test of the
original project draft (2026-07-17) caught a flaw. Read these before changing the design —
they are the "why," and several are the difference between an honest project and one that
dies in a technical screen.

---

## ADR-0001 — Reframe from "benchmark" to "controlled ablation lab"

**Status:** accepted.
**Context:** the original draft positioned this as a "memory evaluation benchmark" whose hook
was "almost nobody measures agent memory." Research falsified that: LongMemEval, LoCoMo, BEAM,
MemBench (ACL 2025), EvoMemBench, GroupMemBench, plus vendor leaderboards from Mem0, Zep, and
Letta all measure agent memory in 2026. A new "benchmark" framed as novel would be dismissed
on sight.
**Decision:** reposition as a *controlled ablation lab*. Those benchmarks rank memory
*systems* on *fixed recall datasets*. memprobe ablates the *policy decisions inside one
system* (write-gate threshold, forgetting policy, retrieval backend) on *parameterized*
scenarios whose difficulty is dial-able. Different question, honestly scoped, doesn't compete
on scale.
**Consequence:** the word "benchmark" is avoided in headline copy; the README explicitly
cites and defers to the leaderboards.

## ADR-0002 — Rename membench -> memprobe

**Status:** accepted.
**Context:** "MemBench" is a published ACL 2025 paper (arXiv 2506.21605) with a public repo.
Reusing the name reads as ignorance or plagiarism.
**Decision:** `memprobe` (verified clear on PyPI and GitHub as of 2026-07). Signals
investigation under control, not a leaderboard.

## ADR-0003 — Build on `langgraph.store`; procedural memory is a stretch goal

**Status:** accepted.
**Context:** LangMem already ships episodic/semantic/procedural memory on top of LangGraph's
`Store`. Hand-rolling those primitives would reinvent `pip install langmem` and impress no
one. Procedural memory (learned behavioral policies) is also the hardest type to implement
*and* to evaluate honestly in a few weekends.
**Decision:** use the raw LangGraph `Store` (`langgraph.store.memory.InMemoryStore`,
`store.put(namespace, key, value)`, `store.search(...)`) as the substrate. Put the
originality in the **policy layer + harness**. Ship episodic + semantic in v1; procedural is
a labeled stretch goal. LangMem may later slot in as a comparison backend.
**Consequence:** the memory modules are thin wrappers over the Store, not new datastores.

## ADR-0004 — "Gated writes" become a confidence threshold τ (an ablation axis)

**Status:** accepted.
**Context:** the draft described "gated writes with human-visible staging." In an *automated*
benchmark harness there is no human to approve a write.
**Decision:** the write gate is a **confidence threshold τ**: extract a candidate fact, write
it only if the extractor's confidence ≥ τ. τ is swept as an ablation axis (0.7 vs 0.9 in the
default matrix). A vague safety feature becomes a measured curve.
**Consequence:** `memory/semantic.py` owns the gate; `write_gate_tau` is a first-class config.

## ADR-0005 — Break the LLM-generates / LLM-grades circularity

**Status:** accepted.
**Context:** if an LLM invents the scenario and an LLM grades the answer, the metric is
circular and a reviewer will say so in the first five minutes.
**Decision:**
- Scenarios are **program-generated**: the generator picks facts and the turns where they are
  introduced/updated/contradicted. An LLM only surface-realizes phrasing (with a templated,
  model-free fallback). Ground truth is the generator's.
- The **primary metric is deterministic** (slot-check the response against known ground truth).
- LLM-as-judge is **secondary**, for response quality only, and every judge number ships with
  a hand-labeled **agreement audit** (%agreement on a sample).
**Consequence:** `scenarios/` and `eval/metrics.py` never call a model for ground truth.

## ADR-0006 — Three mandatory anchors gate every run

**Status:** accepted.
**Context:** a memory eval with no floor/ceiling/placebo is uninterpretable.
**Decision:** every run validates three anchors:
- **calibration:** memory-OFF scores ≈ chance (if not, the task leaks the answer).
- **upper bound:** oracle memory (ground-truth facts injected directly) sets the ceiling.
- **placebo:** shuffled/other-user memory must score no better than memory-OFF (if it does,
  retrieval is leaking across users — run is void).
Report mean ± 95% CI across seeds and users; single-run point numbers are not a result.
**Consequence:** `eval/anchors.py` is not optional; the harness aborts a run whose anchors fail.

## ADR-0007 — Context-policy ablation is the v2 family (not a separate harness project)

**Status:** accepted (2026-07-17); **do not start before v1 ships.**
**Context:** the idea of a standalone "dynamic agentic harness with context layering" was
evaluated. The agent-framework space is saturated (LangGraph/CrewAI/AutoGen/OpenAI Agents
SDK/smolagents/Claude Agent SDK/...), and frameworks are judged by adoption, not existence.
Meanwhile the 2026 context-engineering literature (compaction validation, "governance decay":
compaction silently erasing constraints) is naming a measurement gap with exactly this lab's
shape.
**Decision:** context policies become memprobe's second ablation family, reusing the whole
apparatus: compaction strategy (none/truncate/summarize), context budget, and tool-result
clearing as axes; full-context as the oracle anchor and no-history as the floor; and — the
differentiated metric — *compaction survival*: because ground truth is program-generated, we
know which facts/constraints lived in discarded turns, so "compaction erased something the
agent later needed" is deterministically checkable.
**Consequence:** no new repo; v2 milestones get specced only after the v1 results table
exists.
**Addendum (context-layer talk, Jul 2026):** a further v2 axis — shared vs. siloed stores
across N agents, scored on cross-agent answer consistency (the "sales and finance report two
different revenue numbers" failure, made measurable). Namespace config decides shared vs.
per-agent; the generator knows the true answer; metric = agreement AND correctness across
agents. Nearly free given the existing machinery.

## ADR-0008 — Extraction confidence is evidence-grounded, never model-self-reported

**Status:** accepted (2026-07-18, milestone 2).
**Context:** the write gate (ADR-0004) needs a confidence signal. The obvious source — asking
the extraction model "how confident are you?" — has three problems: it is a constant-ish vibe
on small models, it changes meaning across model backends (so a tau sweep would ablate model
calibration, not the gate), and it can't run on the deterministic stub path at all.
**Decision:** the MODEL proposes candidate (key, value) pairs; the EXTRACTOR assigns
confidence from source-text evidence via a fixed, documented tier table
(`memory/semantic.py`): clean in-vocabulary assertion 0.95 > update/correction assertion 0.85
> off-vocabulary key 0.75 > hedged/out-of-set 0.55 > no supporting assertion 0.30, plus a
small corroboration bonus. Updates are discounted because revisions of a stored belief are
empirically likelier to flip again; off-vocab keys are exactly the "plausible junk" a strict
gate should refuse (the generator's distractors land there by construction).
**Consequence:** the tiers make tau=0.7 vs tau=0.9 a REAL behavioral fork on the stub path
(0.7 tracks revisions but admits junk; 0.9 refuses junk but goes stale on revisions — that
curve is the headline of the tau ablation). A hallucinated extraction has no evidence and
dies at every gate. The definition is constant across model backends, so the sweep measures
the gate.

## ADR-0009 — One canonical context surface for every memory type

**Status:** accepted (2026-07-18, milestone 2).
**Context:** episodic recaps and semantic facts naturally serialize differently; a responder
(especially the deterministic stub) could succeed or fail on FORMAT, confounding the
episodic-vs-semantic comparison.
**Decision:** the respond-time context builder renders every retrieved memory — semantic
record or episodic recap pair — as the same assertion line `my <key> is <value>`, ordered
oldest -> newest (so "later = more current" is encoded honestly; the stub literally keeps the
last value per key). Episode summaries are a CONTRACT ('key: value' pairs) that the builder
unpacks into that surface.
**Consequence:** memory types differ only in WHAT was stored/retrieved (gating, granularity,
recency windows), never in how it was punctuated. The ablation measures policy, not parsing.

## ADR-0010 — Respond context is memory-only (no live-session turns) in v1

**Status:** accepted (2026-07-18, milestone 2).
**Context:** a real deployed agent sees the live conversation plus memory. But if probes can
be answered from the live session, memory_off scores above zero on exactly the probes whose
facts changed in the probe session, the calibration floor drifts above chance, and the
placebo anchor inherits the same lift — all three anchors get muddier for a marginal gain in
realism.
**Decision:** the respond node sees retrieved memory and the probe question only. The
graph's per-session order (retrieve -> respond -> extract -> gated_write) means a fact
updated in the probe session is not yet written when the probe is answered — every memory
policy misses it, only oracle gets it right.
**Consequence:** anchors stay clean (memory_off = pure refusal floor). The oracle-vs-best-
config gap includes same-session updates and the report says so. Live-context interplay is
not lost — it is the v2 context-policy family's opening axis (ADR-0007), where it can be
measured instead of assumed.

## ADR-0003 addendum (milestone 2) — how the LangGraph adapter preserves lab semantics

Two adapter choices in `memory/store.py::LangGraphStore`: (1) item keys carry a monotone
sequence suffix because LangGraph's `put()` replaces by key while the lab requires APPEND
history (a superseded value must stay observable or staleness/forgetting have nothing to
measure); (2) `search` re-implements DictStore's transparent direct-read ranking (substring
key match, most-recent first) instead of delegating to substrate text search, so both
backends rank identically and the direct-vs-embedding ablation measures the retrieval
policy, not two vendors' rankers. Parity is test-enforced (`test_langgraph_store.py`).

## Open questions (resolve during build)
- Does the direct-vs-embedding crossover even appear at n_users=20 scale, or must scenarios
  grow to make retrieval matter? (PKB's finding suggests direct wins while small — that would
  itself be the honest result to report.)
- Minimum seeds for the CIs to separate configs? Start at 3, widen if intervals overlap.
- Is templated (model-free) surface realization *too* easy to memorize, inflating scores?
  Check memory-OFF calibration on templated vs LLM-realized scenarios.
