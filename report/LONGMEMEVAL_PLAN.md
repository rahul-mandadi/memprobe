# Pre-registration: external validity on a LongMemEval slice

Written 2026-09-25, before any LongMemEval data was downloaded or any result existed.
Commit this file before the run so its timestamp predates the numbers.

## Question

The synthetic matrix found that extracted semantic facts beat episodic session recaps (0.84 vs
0.66 task success) and that embedding retrieval tied direct lookup. Do those two results hold
on real long-horizon chat histories that memprobe did not generate?

## Data

LongMemEval (arXiv:2410.10813), the maintained cleaned release on Hugging Face
(`xiaowu0162/longmemeval-cleaned`): `longmemeval_s_cleaned.json` (277 MB; 500 questions, each
with roughly 115K tokens of multi-session history) and `longmemeval_oracle.json` (15 MB; the
same questions with only the evidence sessions). Check the dataset licence before use.

**Slice:** 60 questions, stratified across the benchmark's question types, with the
knowledge-update type over-sampled because it is the real-data counterpart of memprobe's
contradiction axis. The seed for the sample is fixed in the config before the run.

## What is compared

Four configs, reusing the existing policies: episodic recaps only; semantic facts with write
gate 0.9 and direct lookup; semantic facts with embedding retrieval (MiniLM); and the two
anchors below. The write-gate and forgetting axes are dropped: they did not separate on
synthetic data, and this run exists to test the effects that did.

**Anchors, same logic as the synthetic lab:**
- memory off: the question with no history. Must score near the floor.
- oracle: the evidence sessions from `longmemeval_oracle.json` placed directly in context. The ceiling.
- placebo: another question's history. Must not beat memory off by more than coincidence.

## Metric

**Headline: deterministic.** Normalised containment of the gold answer in the response, the
same rule as memprobe's slot-check. Scored only on questions whose gold answer is a short
span (an entity, a number, a date); the share of the slice this excludes is reported, never
hidden. Abstention questions are scored as correct when the response declines.

The benchmark's own evaluation uses an LLM judge. That is **not** the headline here. If a judge
is run, it is secondary and ships with a hand-labelled agreement audit on at least 30
responses, as the lab already requires.

## Pre-registered outcomes

- **Kill condition:** if the oracle anchor scores below 0.60 under the deterministic metric,
  the metric cannot read LongMemEval answers. Stop, report that, and publish no config numbers.
- **Void condition:** any anchor failing voids the run, exactly as on synthetic data.
- **Reportable either way:** semantic over episodic is the hypothesis. If it does not separate,
  that is a negative external-validity result and gets reported as such.

## Cost and time

Extraction reads each history once per memory type and is shared across configs:
60 x ~115K tokens x 2 memory types is about 14M input tokens, about $14 on Haiku 4.5, plus
answering. **Budget cap: $25.** A 5-question pilot runs first; if its measured cost projects
past the cap, the slice shrinks before the full run, not after.

**Hard stop: 2026-10-02.** If the adapter and the run are not done by then, the write-up says
what exists and the synthetic results stand alone.
