# memprobe results — run `20260925-144428`

- agent model: `bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0` · judge: `stub` · embeddings: `local:all-MiniLM-L6-v2`
- scenarios: 20 users × 8 sessions × 6 turns; contradiction 0.25, distractors 0.3; seeds [0, 1, 2]
- every interval: mean [lo, hi] at 95% confidence across users × seeds (n = 60 per config)

## Anchors (credibility gates)

- chance floor (exact, from closed value sets): **0.300**
- memory OFF (must not exceed chance + tolerance): task success 0.067 [0.022, 0.111]
- oracle (ground truth injected; the ceiling): task success 1.000 [1.000, 1.000]
- shuffled other-user memory (must not beat coincidence): task success 0.308 [0.219, 0.398]

**Gate result: PASS**
- anchors OK: calibration, upper bound, and placebo all hold.

## Ablation matrix

| config | task success | contamination | stale answers | tokens/session | est. $/run |
|---|---|---|---|---|---|
| memory_off ⚓calibration | 0.067 [0.022, 0.111] | 0.100 [0.048, 0.152] | 0.058 [0.017, 0.100] | 38.66 | $0.0625 |
| oracle ⚓upper_bound | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 25.29 | $0.0197 |
| shuffled_placebo ⚓placebo | 0.308 [0.219, 0.398] | 0.458 [0.362, 0.554] | 0.358 [0.266, 0.451] | 145.57 | $0.1467 |
| episodic_only | 0.658 [0.568, 0.749] | 0.317 [0.222, 0.412] | 0.150 [0.077, 0.223] | 127.65 | $0.0987 |
| semantic_gate_07 | 0.825 [0.750, 0.900] | 0.375 [0.284, 0.466] | 0.158 [0.085, 0.232] | 242.03 | $0.2010 |
| semantic_gate_09 | 0.842 [0.768, 0.915] | 0.358 [0.266, 0.451] | 0.142 [0.070, 0.213] | 240.71 | $0.1996 |
| semantic_embedding | 0.842 [0.768, 0.915] | 0.358 [0.260, 0.457] | 0.142 [0.070, 0.213] | 246.29 (+686 embed) | $0.2012 |
| semantic_forget_decay | 0.767 [0.686, 0.847] | 0.300 [0.210, 0.390] | 0.150 [0.077, 0.223] | 241.46 | $0.2003 |

## Accuracy vs cost (Pareto)

Pareto-efficient study configs: **episodic_only, semantic_gate_09**

```
task success ↑ (1.0 top) vs tokens/session → (cheap left)

  |                                            
  |                                        D  E
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
  A = episodic_only (tokens/session 127.7, success 0.658)
  B = semantic_gate_09 (tokens/session 240.7, success 0.842)
  C = semantic_forget_decay (tokens/session 241.5, success 0.767)
  D = semantic_gate_07 (tokens/session 242.0, success 0.825)
  E = semantic_embedding (tokens/session 247.7, success 0.842)
```

## Response quality (LLM-judge, secondary)

Not run (judge model: `stub`). Judge numbers may only appear together with a hand-labeled agreement audit (ADR-0005); the deterministic stub cannot produce an auditable judgment, so this section is intentionally empty rather than decorative.

## Reading these numbers

- Two configs differ only if their intervals separate (eval/stats.separated) — overlapping CIs are a tie, whatever the means say.
- Contamination counts responses asserting a value with no support in the user's own history (other users' current values); the statistical cross-user leak gate is the placebo anchor (ADR-0013).
- Stale answers are the flagged-AND-wrong subset of the same must_not_contain channel (superseded own values and other-user values share it in v1 — metrics.py says so out loud; a per-user unique-token tracer is the sharper v2 instrument). For the placebo column that makes 'stale' read as 'served a neighbor's value and missed'.
- The oracle-vs-best-config gap includes same-session updates, which no memory policy can answer under the memory-only respond context (ADR-0010).
