# memprobe results — run `20260718-010457`

- agent model: `stub` · judge: `stub` · embeddings: `hash`
- scenarios: 20 users × 8 sessions × 6 turns; contradiction 0.25, distractors 0.3; seeds [0, 1, 2]
- every interval: mean [lo, hi] at 95% confidence across users × seeds (n = 60 per config)

> **Stub-model run.** The agent is the deterministic stub (a harness driver, not a real model): these numbers validate the harness, anchors, and policy mechanics — treat absolute values as harness properties, not model quality.

## Anchors (credibility gates)

- chance floor (exact, from closed value sets): **0.300**
- memory OFF (must not exceed chance + tolerance): task success 0.000 [0.000, 0.000]
- oracle (ground truth injected; the ceiling): task success 1.000 [1.000, 1.000]
- shuffled other-user memory (must not beat coincidence): task success 0.275 [0.191, 0.359]

**Gate result: PASS**
- anchors OK: calibration, upper bound, and placebo all hold.

## Ablation matrix

| config | task success | contamination | stale answers | tokens/session | est. $/run |
|---|---|---|---|---|---|
| memory_off ⚓calibration | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 12.5 | $0.0000 |
| oracle ⚓upper_bound | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 15.5 | $0.0000 |
| shuffled_placebo ⚓placebo | 0.275 [0.191, 0.359] | 0.392 [0.306, 0.478] | 0.333 [0.242, 0.424] | 76.11 | $0.0000 |
| episodic_only | 0.658 [0.568, 0.749] | 0.242 [0.161, 0.322] | 0.150 [0.077, 0.223] | 71.95 | $0.0000 |
| semantic_gate_07 | 0.867 [0.796, 0.937] | 0.333 [0.242, 0.424] | 0.133 [0.063, 0.204] | 136.04 | $0.0000 |
| semantic_gate_09 | 0.858 [0.787, 0.930] | 0.267 [0.183, 0.351] | 0.142 [0.070, 0.213] | 135.53 | $0.0000 |
| semantic_embedding | 0.867 [0.796, 0.937] | 0.333 [0.242, 0.424] | 0.133 [0.063, 0.204] | 138.9 (+686 embed) | $0.0000 |
| semantic_forget_decay | 0.783 [0.703, 0.864] | 0.292 [0.199, 0.385] | 0.142 [0.070, 0.213] | 135.78 | $0.0000 |

## Accuracy vs cost (Pareto)

Pareto-efficient study configs: **episodic_only, semantic_gate_07, semantic_gate_09**

```
task success ↑ (1.0 top) vs tokens/session → (cheap left)

  |                                            
  |                                       BD  E
  |                                        C   
  |A                                           
  |                                            
  |                                            
  |                                            
  |                                            
  |                                            
  |                                            
  |                                            
  |                                            
  +--------------------------------------------
  A = episodic_only (tokens/session 72.0, success 0.658)
  B = semantic_gate_09 (tokens/session 135.5, success 0.858)
  C = semantic_forget_decay (tokens/session 135.8, success 0.783)
  D = semantic_gate_07 (tokens/session 136.0, success 0.867)
  E = semantic_embedding (tokens/session 140.3, success 0.867)
```

![accuracy vs cost](pareto.png)

## Response quality (LLM-judge, secondary)

Not run (judge model: `stub`). Judge numbers may only appear together with a hand-labeled agreement audit (ADR-0005); the deterministic stub cannot produce an auditable judgment, so this section is intentionally empty rather than decorative.

## Reading these numbers

- Two configs differ only if their intervals separate (eval/stats.separated) — overlapping CIs are a tie, whatever the means say.
- Contamination counts responses asserting a value with no support in the user's own history (other users' current values); the statistical cross-user leak gate is the placebo anchor (ADR-0013).
- Stale answers are the flagged-AND-wrong subset of the same must_not_contain channel (superseded own values and other-user values share it in v1 — metrics.py says so out loud; a per-user unique-token tracer is the sharper v2 instrument). For the placebo column that makes 'stale' read as 'served a neighbor's value and missed'.
- The oracle-vs-best-config gap includes same-session updates, which no memory policy can answer under the memory-only respond context (ADR-0010).
