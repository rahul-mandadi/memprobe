"""The LangGraph agent: retrieve -> respond -> extract -> gated-write.

Wiring (NOTES.md ADR-0003/0004): a LangGraph `StateGraph` with memory operations as
first-class nodes over the `Store`. Per turn:
  1. retrieve  — pull candidate memories for the user (direct or embedding backend)
  2. respond   — answer the user given retrieved memory (the response scored by metrics.py)
  3. extract   — pull candidate facts from the exchange (semantic.extract_facts)
  4. gated_write — write only facts with confidence >= tau (semantic.apply_write_gate)

The graph is assembled from a PolicyConfig so every ablation cell is the SAME graph with
different knobs — that is what makes the comparison clean. Stubbed: needs models + the real
Store adapter. The model-free path used by tests drives nodes directly (see harness).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyConfig:
    """Resolved knobs for one ablation cell (built from config.MemConfig)."""

    use_episodic: bool = False
    use_semantic: bool = False
    retrieval: str = "direct"          # "direct" | "embedding"
    write_gate_tau: float = 0.7
    forgetting: str | None = None
    half_life_days: float = 30.0
    oracle: bool = False               # inject ground truth directly (upper-bound anchor)
    memory_off: bool = False           # no memory at all (calibration anchor)
    scramble_namespaces: bool = False  # serve another user's memory (placebo anchor)


def build_agent(policy: PolicyConfig, store, agent_model, embedder=None):
    """Compile a LangGraph agent for one policy cell. TODO(milestone-2).

    Returns a runnable graph exposing `.run_session(user_id, session)` -> responses. Keep node
    boundaries clean so tracing (agent/tracing scope) can attribute tokens per node."""
    raise NotImplementedError(
        "TODO(milestone-2): assemble StateGraph(retrieve, respond, extract, gated_write) "
        "over langgraph.store; branch retrieval on policy.retrieval; apply write gate"
    )
