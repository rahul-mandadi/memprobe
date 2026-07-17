"""memprobe — a controlled ablation lab for agent-memory policy decisions.

Not a benchmark (see notebooks/NOTES.md ADR-0001). A sandbox that ablates the policy knobs
inside a single memory system — write-gate threshold, forgetting policy, retrieval backend —
on parameterized synthetic scenarios with known ground truth, reporting the
accuracy / cost / contamination trade-off with confidence intervals.
"""

__version__ = "0.1.0.dev0"
