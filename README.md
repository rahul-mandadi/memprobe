# memprobe

**A controlled ablation lab for agent-memory *policy* decisions.**

Existing memory benchmarks ([LongMemEval](https://arxiv.org/abs/2410.10813),
[LoCoMo](https://arxiv.org/abs/2402.17753),
[MemBench](https://arxiv.org/abs/2506.21605)) rank memory *systems* on *fixed recall
datasets* — "how accurately can system X answer questions about past sessions." That's a
leaderboard question. memprobe asks a different, practitioner-facing question the
leaderboards don't:

> Given one memory system, **which policy knobs actually matter**, and what do they cost?
> Is it worth gating writes at confidence 0.7 vs 0.9? Does a forgetting half-life help or
> just lose information? Where does embedding retrieval start beating a direct structured
> read — and how many tokens does each configuration burn to get its accuracy?

memprobe generates **parameterized** synthetic multi-session scenarios (you dial the
contradiction rate, session count, and distractor density; ground truth is known because the
generator injected it), runs an agent through them under a **matrix of memory policies**, and
reports the **accuracy / cost / contamination trade-off** with confidence intervals.

It is **not** a competitor to LongMemEval — it doesn't rank vendors and it runs at small,
cheap scale on purpose. It's a sandbox for the engineering trade-offs *inside* a memory
system.

> **Status: v1 complete.** The full ablation matrix runs end-to-end, hermetically
> (`make bench`: no model, no network, no keys — deterministic stub agent + hashing
> embedder), anchors validate, and the table below is **measured**, not illustrative.
> Real-model backends (Ollama / Anthropic / MiniLM) are config swaps. 123 tests, zero
> xfails. Full report: [`report/out/results.md`](report/out/results.md).

---

## The result on a real model (2026-09-25)

Claude Haiku 4.5 as the agent (via AWS Bedrock), `all-MiniLM-L6-v2` embeddings, the same
program-generated scenarios: 20 users x 8 sessions x 6 turns, 3 seeds, n = 60 per config,
95% CIs. Anchors pass: memory off 0.067 (chance 0.300), oracle 1.000, shuffled placebo 0.308.

| Config | Task success | Tokens/session |
|---|---|---|
| episodic recaps only | 0.658 [0.568, 0.749] | 127.7 |
| + semantic facts, write gate 0.7 | 0.825 [0.750, 0.900] | 242.0 |
| + semantic facts, write gate 0.9 | 0.842 [0.768, 0.915] | 240.7 |
| semantic, embedding retrieval | 0.842 [0.768, 0.915] | 246.3 (+686 embedding) |
| semantic + recency-decay forgetting | 0.767 [0.686, 0.847] | 241.5 |

- **Extracted semantic facts beat episodic recaps** (0.84 vs 0.66, intervals separated).
- **Embedding retrieval tied direct lookup** and paid 686 extra embedding tokens.
- **The write-gate threshold and forgetting did not separate** from the baseline semantic config.
- **The stand-in run's ordering replicated** with a real LLM answering.

Full matrix for about $1.13. What it does not show, including why the contamination column
should not be read as cross-user leakage yet: [`report/FINDINGS.md`](report/FINDINGS.md).

## The harness validation run (v1, hermetic stub-model run, 2026-07-18)

_Measured by `make bench` on program-generated scenarios with known ground truth: 20 users
× 8 sessions × 6 turns, contradiction 0.25, distractors 0.3; every cell is mean [lo, hi] at
95% confidence across users × seeds (n = 60). **Stub-model run:** the agent is the
deterministic harness driver, so these numbers validate the harness + policy mechanics —
they are the lab working, not a model's quality._

| Config | Task success | Contamination | Stale-answer | Tokens/session | $/full run |
|---|---|---|---|---|---|
| memory OFF (calibration floor) | 0.000 [0.000, 0.000] | — | — | 12.5 | $0 |
| oracle memory (upper bound) | 1.000 [1.000, 1.000] | 0.000 | 0.000 | 15.5 | $0 |
| shuffled-memory placebo | 0.275 [0.191, 0.359] | 0.392 [0.306, 0.478] | — | 76.1 | $0 |
| episodic only | 0.658 [0.568, 0.749] | 0.242 [0.161, 0.322] | 0.150 [0.077, 0.223] | 72.0 | $0 |
| + semantic (write-gate τ=0.7) | 0.867 [0.796, 0.937] | 0.333 [0.242, 0.424] | 0.133 [0.063, 0.204] | 136.0 | $0 |
| + semantic (write-gate τ=0.9) | 0.858 [0.787, 0.930] | 0.267 [0.183, 0.351] | 0.142 [0.070, 0.213] | 135.5 | $0 |
| semantic, embedding retrieval | 0.867 [0.796, 0.937] | 0.333 [0.242, 0.424] | 0.133 [0.063, 0.204] | 138.9 (+686 embed) | $0 |
| semantic + recency-decay forgetting | 0.783 [0.703, 0.864] | 0.292 [0.199, 0.385] | 0.142 [0.070, 0.213] | 135.8 | $0 |

What the stub run already shows (claims phrased by CI separation, nothing else):

- **Anchors hold:** memory-OFF sits at the refusal floor (below the exact 0.300 chance
  floor), oracle at the ceiling, and the placebo lands at coincidence level (≈ chance) —
  no cross-user leakage (see ADR-0013 for why placebo compares against *chance*, not the
  refusing floor).
- **Gated semantic facts beat whole-session recaps** (0.867 vs 0.658, intervals separated):
  episodic-only loses facts that fall out of its recency window.
- **The τ trade-off is directional, not yet separated at 3 seeds:** τ=0.9 shows lower
  contamination (0.267 vs 0.333) at slightly lower success — the designed
  strictness-vs-coverage curve, but the intervals overlap, so v1 reports a trend and the
  seed count it would take to resolve it, not a conclusion.
- **The direct read held up at this scale** (the PKB-sequel question): embedding retrieval
  matched direct exactly (0.867) while paying 686 extra embedding tokens — at 20-user
  scale, structured key lookup is Pareto-dominant. That's the honest finding, not a failure.
- **Forgetting cost success without buying staleness** at contradiction 0.25 (0.783 vs
  0.867, directional): the decay dropped old-but-still-current facts.

The **shuffled-memory placebo** and **oracle** rows are not decoration — they are the
credibility anchors (see [Why you can trust the numbers](#why-you-can-trust-the-numbers)).
The placebo's dashes: its "stale" column reads as "served a neighbor's value and missed"
under the shared must_not_contain channel, so v1 reports it only in the full report with
that caveat spelled out.

---

## Reference use case (swappable)

A **customer-support agent** for a synthetic SaaS product that remembers users across
sessions — the memory deployment companies actually ship, with vivid, checkable failure
modes: stale preferences, a fact contradicted in a later ticket, another user's detail
surfacing (contamination), or a bloated context from over-retrieval. The domain lives behind
one module (`usecase/`); swap it without touching the harness.

## Why you can trust the numbers

A memory eval where an LLM writes the test and an LLM grades it is circular. memprobe breaks
the loop deliberately:

- **Scenarios are program-generated.** Facts, and the turns where they're introduced,
  updated, or contradicted, are chosen by code. An LLM only *surface-realizes* the phrasing
  (and a fully templated, model-free fallback exists so the suite runs with no model at all).
  **Ground truth belongs to the generator, never to a model.**
- **The primary metric is deterministic.** Task success is a slot-check: did the agent's
  response carry/act on the correct seeded fact? No judge needed for the headline number.
- **LLM-as-judge is secondary and audited.** Used only for response *quality*, and every
  reported judge number ships with a hand-labeled **agreement audit** (%agreement on a
  sampled subset). A judge you haven't audited is a vibe, not a metric.
- **Three anchors gate every run:** memory-OFF must score ≈ chance (calibration), **oracle
  memory** (ground-truth facts injected directly) sets the upper bound, and a
  **shuffled/other-user memory** placebo must score no better than chance (ADR-0013); if it
  clears that ceiling, the harness is leaking and the run is void.
- **Multiple seeds, reported with CIs.** Point estimates from one run are not a result.

## Design decisions (the honest ones)

- **Built on `langgraph.store`, not reinvented.** [LangMem](https://github.com/langchain-ai/langmem)
  already ships episodic/semantic/procedural memory. memprobe uses the raw LangGraph `Store`
  as its substrate and puts its originality in the **policy layer + harness**, not in
  re-implementing memory primitives. LangMem can slot in as a comparison backend later.
- **"Gated writes" became a measured threshold.** In an automated harness there is no human
  to approve a write, so the gate is a **confidence threshold τ** — and τ is an *ablation
  axis*, not a feature. That's strictly better: a vague safety story turns into a curve.
- **Procedural memory is a stretch goal, not v1.** It's the hardest type to implement *and*
  to evaluate honestly in a few weekends. v1 ships episodic + semantic (both first-class in
  the Store, both cleanly measurable). See `notebooks/NOTES.md` ADR-0003.
- **Small and cheap on purpose.** LongMemEval scenarios run 115K–1.5M tokens *each*. memprobe
  runs tiny synthetic scenarios so the full matrix fits in a weekend and under ~$20 (or $0
  fully local). It buys *controlled ablation*, not scale — and says so.

## Build order

Each milestone left a runnable artifact; the deterministic core (scenarios + metrics +
anchors) came first because it is the credibility foundation. All five v1 milestones are
**shipped** (the strict-xfail worklist in `tests/test_pipeline_stub.py` flipped test by
test — its git history is the build history):

1. ✅ **Scenario generator + deterministic metrics + anchors.** memory-OFF ≈ chance, oracle =
   ceiling, shuffled = placebo.
2. ✅ **Episodic + semantic memory on the Store**, write-gate τ, direct read.
3. ✅ **Embedding retrieval backend** (hermetic hashing encoder by default; local
   `all-MiniLM-L6-v2` via config) → the direct-vs-embedding comparison (the PKB sequel).
4. ✅ **Forgetting ablation** + LLM-judge with the agreement audit (judge numbers cannot
   render without the audit).
5. ✅ **Full config matrix + results-first report** (table above + accuracy/cost Pareto).
- **Stretch (open):** procedural memory; anchor difficulty against a LongMemEval-S slice for
  external validity; LangMem as a drop-in backend; v2 = context-policy family (ADR-0007).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + tests, no model needed
make test                      # deterministic suite (123 tests) — no API, no Ollama
make bench                     # full hermetic matrix → data/runs/<ts>/ + report/out/
# optional extras:
pip install -e ".[plot]"       # pareto.png in the report (ASCII Pareto works without it)
pip install -e ".[local]"      # sentence-transformers + ollama client for real-model runs
pip install -e ".[api]"        # anthropic client for a stronger judge
```

**Model strategy (default = free & offline):** agent + judge default to a local Ollama model;
embeddings default to local `sentence-transformers/all-MiniLM-L6-v2` (22 MB). A deterministic
**stub model** backs the test suite so CI needs no keys and no Ollama. Anthropic (Haiku 4.5,
~$1/$5 per M tokens → ≈$8 for a full matrix) is opt-in for a higher-quality judge.

## Positioning, stated plainly

memprobe does not claim to beat, replace, or rank alongside LongMemEval, LoCoMo, or MemBench.
Those are curated recall leaderboards for memory *systems*. memprobe is a controlled sandbox
for the *policy* trade-offs a practitioner tunes inside one system, on scenarios whose
difficulty you can dial. Different question, smaller scale, honestly scoped.

## License

MIT
