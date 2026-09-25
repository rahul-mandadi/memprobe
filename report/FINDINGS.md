# memprobe findings: agent-memory policy on a real model

Run `20260925-144428`, 2026-09-25. Agent: Claude Haiku 4.5 (via AWS Bedrock). Embeddings:
`all-MiniLM-L6-v2`. Judge: not run. Full table: [`out/bedrock-haiku-4-5/results.md`](out/bedrock-haiku-4-5/results.md).

## The two problems

**Domain.** An assistant that remembers users across sessions can fail in ways a single-turn
eval never sees: it answers with a preference the user has since changed, it surfaces another
user's detail, or it drags so much history into context that every turn gets expensive. Teams
tune memory settings (what to store, when to forget, how to retrieve) with little evidence
about which of those settings actually move the outcome.

**Analytic.** On multi-session scenarios with known ground truth, measure how task success,
wrong-value answers and token cost respond to four policy choices: episodic recaps versus
extracted semantic facts, the confidence threshold for writing a fact, direct structured
lookup versus embedding retrieval, and recency-decay forgetting.

## Who the result is for

No outside reviewer. The honest substitute is a real decision the result feeds: the
author's personal knowledge base deliberately stores facts for direct lookup and does not use
a vector store. The direct-versus-embedding comparison below tests that choice.

## What was compared, and how

- **Scenarios:** program-generated. 20 synthetic users x 8 sessions x 6 turns, 25% of facts
  contradicted in a later session, 30% distractor density, 3 seeds. The generator owns the
  ground truth; no model writes or grades the test.
- **Headline metric:** task success, a deterministic check that the response carries the
  current seeded value. No judge.
- **Anchors, fixed before any result:** memory off must not exceed chance, an oracle (ground
  truth injected) sets the ceiling, and another user's shuffled memory must not beat
  coincidence. A run whose anchors fail is void.
- **Intervals:** 95% CIs across users x seeds, n = 60 per config. A difference is called only
  when intervals separate.

## Results

Anchors pass: memory off 0.067 (chance floor 0.300), oracle 1.000, shuffled placebo 0.308.

| Config | Task success | Tokens/session |
|---|---|---|
| episodic recaps only | 0.658 [0.568, 0.749] | 127.7 |
| + semantic facts, write gate 0.7 | 0.825 [0.750, 0.900] | 242.0 |
| + semantic facts, write gate 0.9 | 0.842 [0.768, 0.915] | 240.7 |
| semantic, embedding retrieval | 0.842 [0.768, 0.915] | 246.3 (+686 embedding) |
| semantic + recency-decay forgetting | 0.767 [0.686, 0.847] | 241.5 |

1. **Extracted semantic facts beat episodic recaps**, 0.84 against 0.66, intervals separated.
   Recaps lose facts that fall out of the recency window.
2. **Embedding retrieval bought nothing at this scale.** It tied direct lookup exactly and
   paid 686 extra embedding tokens. The direct-lookup choice holds here.
3. **The write-gate threshold did not matter** within this range (0.825 against 0.842,
   overlapping).
4. **Forgetting looks harmful but is not established** (0.767 against 0.825, overlapping). The
   decay drops facts that are old but still current.
5. **The real model reproduced the stand-in run's ordering.** The July harness run with a
   deterministic stand-in agent gave semantic 0.867 and episodic 0.658. The same conclusions
   hold with a real LLM answering.

Cost of the full real-model matrix: about $1.13 on Bedrock, 70 minutes.

## What this does not show

- **Synthetic data only.** Twenty users with closed-vocabulary facts. Nothing here says the
  effects hold on real conversation logs or at production scale.
- **Templated scenarios.** The phrasing came from templates, not from a model paraphrasing
  them, so retrieval had exact token overlap to work with. That favours direct lookup; the
  embedding result may change on paraphrased text. The realization step exists but was not
  used in this run.
- **One model.** Haiku 4.5 only.
- **The wrong-value column is not cross-user leakage.** The results table labels it
  contamination, but it counts any response containing any disallowed value: another user's,
  the user's own superseded value, or a coincidental hit from a guess. That is why memory off
  scores 0.10 on it. Read task success, not that column, until the two sources are scored
  separately.
- **One anchor rule changed after integration.** The placebo originally had to stay below the
  memory-off floor. A model that refuses on empty memory puts that floor at zero, which made
  honest coincidence look like leakage and voided every run. The rule now compares against
  chance (ADR-0013, `notebooks/NOTES.md`). It is principled, and it was decided after seeing
  runs, so it is stated here.
- **Response quality was not judged.** The judge is audited by design and was left off.

## Next

- Score other-user and own-stale values separately, then re-read the wrong-value columns.
- Run the matrix on paraphrased scenarios to test the embedding result where token overlap
  stops helping.
- External validity: a small slice of the public LongMemEval benchmark, to test whether the
  semantic-over-episodic effect holds on real long-horizon chat histories.

## Contributions

Research question, experiment design, anchors, analysis and write-up: Rahul Reddy Mandadi.
