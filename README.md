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

> **Status: scaffold.** Structure, typed stubs, design decisions, and the deterministic
> test core are in place; model-dependent pieces are stubbed. See [Build order](#build-order).
> Numbers below are **illustrative placeholders** until the harness runs — this README is
> results-first by design, so the table is wired up before the data exists.

---

## The result this produces (illustrative — NOT yet measured)

_Once built, `make bench` fills this table. Every cell is measured on generated scenarios
with known ground truth; ± is a 95% CI across users and seeds._

| Config | Task success | Contamination | Stale-answer | Tokens/session | $/full run |
|---|---|---|---|---|---|
| memory OFF (calibration floor) | _~chance_ | — | — | low | — |
| oracle memory (upper bound) | _~ceiling_ | 0 | 0 | — | — |
| episodic only | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| + semantic (write-gate τ=0.7) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| + semantic (write-gate τ=0.9) | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| semantic, embedding retrieval | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| semantic + recency-decay forgetting | _tbd_ | _tbd_ | _tbd_ | _tbd_ | _tbd_ |

The **shuffled-memory placebo** and **oracle** rows are not decoration — they are the
credibility anchors (see [Why you can trust the numbers](#why-you-can-trust-the-numbers)).

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
  **shuffled/other-user memory** placebo must score no better than memory-OFF — if it does,
  the harness is leaking and the run is void.
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

Each milestone leaves a runnable artifact; the deterministic core (scenarios + metrics +
anchors) is milestone 1 because it is the credibility foundation.

1. **Scenario generator + deterministic metrics + anchors.** memory-OFF ≈ chance, oracle =
   ceiling, shuffled = placebo. _This milestone alone is a credible artifact._
2. **Episodic + semantic memory on the Store**, write-gate τ, direct read → first bench column.
3. **Embedding retrieval backend** (local `all-MiniLM-L6-v2`) → the direct-vs-embedding
   comparison (the PKB sequel).
4. **Contamination / staleness / forgetting ablations** + LLM-judge with the agreement audit.
5. **Full config matrix + results-first report** (table above + accuracy/cost Pareto plot).
- **Stretch:** procedural memory; anchor difficulty against a LongMemEval-S slice for
  external validity; LangMem as a drop-in backend.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + tests, no model needed
make test                      # deterministic suite runs with the stub model, no API/Ollama
# optional model backends:
pip install -e ".[local]"      # sentence-transformers + ollama client
pip install -e ".[api]"        # anthropic client for a stronger judge
make bench                     # runs the matrix → report/
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
