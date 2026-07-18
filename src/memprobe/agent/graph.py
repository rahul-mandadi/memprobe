"""The LangGraph agent: retrieve -> respond -> extract -> gated-write.

Wiring (NOTES.md ADR-0003/0004): a LangGraph `StateGraph` with memory operations as
first-class nodes over the `Store`. Per session:
  1. retrieve  — pull candidate memories for the user (direct or embedding backend)
  2. respond   — answer each probe given retrieved memory (responses scored by metrics.py)
  3. extract   — pull candidate facts from the session (semantic.extract_facts)
  4. gated_write — write only facts with confidence >= tau (semantic.apply_write_gate),
                   and/or the episodic session summary

The graph is assembled from a PolicyConfig so every ablation cell is the SAME compiled graph
with different knobs — that is what makes the comparison clean. Anchor semantics live in the
knobs: `oracle` injects ground-truth facts as context, `memory_off` retrieves nothing, and
`scramble_namespaces` redirects READS to another user's namespace while writes stay in the
user's own (every store must be populated for a neighbor to read — see harness).

Two design decisions recorded as ADRs:
- ADR-0009 (canonical context surface): every retrieved memory — semantic fact or episodic
  recap pair — renders into the same "my <key> is <value>" assertion line, ordered oldest to
  newest. The respond model cannot distinguish memory backends by format, so the ablation
  measures what was stored/retrieved, not how it was punctuated.
- ADR-0010 (memory-only respond context): the respond prompt contains retrieved memory and
  the probe question, NOT the current session's live turns. Including live turns would let
  every config (memory_off included) answer probes about facts updated in the probe session,
  lifting the calibration floor above chance and muddying all three anchors. The cost — no
  config can answer a same-session update, so even the best policy sits below oracle — is
  honest and reported. Live-context interplay is exactly the v2 context-policy family
  (ADR-0007).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from memprobe.memory import episodic as _episodic
from memprobe.memory import semantic as _semantic
from memprobe.memory.forgetting import apply_forgetting
from memprobe.memory.retrieval import DirectRetriever
from memprobe.memory.store import MemoryRecord, MemoryStore
from memprobe.scenarios.schema import Probe, Turn
from memprobe.usecase.support import SYSTEM_PROMPT, probe_question


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
    retrieve_k: int = 5                # top-k memories per probe (both backends, fairness)


class AgentState(TypedDict, total=False):
    """State flowing through one session's graph invocation."""

    user_id: str                       # whose conversation this is (writes go here)
    memory_user_id: str                # namespace READS come from (differs under scramble)
    session_index: int
    now: float                         # logical time (harness supplies the clock)
    turns: list[Turn]
    probes: list[Probe]
    oracle_facts: dict[str, str]       # key -> current value (upper-bound anchor only)
    contexts: dict[str, list[str]]     # probe_id -> rendered memory lines
    responses: list[dict[str, str]]    # [{"probe_id", "response"}]
    candidates: list[Any]              # extracted CandidateFacts
    written: int                       # semantic records admitted by the gate


def _render_records(records: list[MemoryRecord]) -> list[str]:
    """Serialize memories into the canonical fact surface (ADR-0009), oldest -> newest.

    Oldest-first matters: the responder treats later assertions as more current (the stub
    literally keeps the last value per key), so rendering order encodes recency honestly.
    Episodic recaps are unpacked into per-fact lines so all memory types share one surface.
    """
    import re

    lines: list[str] = []
    for r in sorted(records, key=lambda r: (r.written_at, r.source_session)):
        if r.key.startswith(_episodic.EPISODE_KEY_PREFIX + ":"):
            for k, v in re.findall(r"\b([\w]+)\s*:\s*([\w]+)\b", r.value.lower()):
                lines.append(f"my {k} is {v}")
        else:
            lines.append(f"my {r.key} is {r.value}")
    return lines


class Agent:
    """A compiled policy cell: one LangGraph graph + its knobs, store, model, retriever.

    The retriever is exposed because it carries the embedding cost meter (`embed_tokens`) —
    the harness reads it for the accuracy-vs-cost accounting (ADR-0011)."""

    def __init__(
        self, graph, policy: PolicyConfig, store: MemoryStore | None, agent_model, retriever=None
    ) -> None:
        self._graph = graph
        self.policy = policy
        self.store = store
        self.model = agent_model
        self.retriever = retriever

    def run_session(
        self,
        user_id: str,
        session,
        *,
        now: float | None = None,
        memory_user_id: str | None = None,
        oracle_facts: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        """Run one session through retrieve->respond->extract->gated_write.

        Returns [{"probe_id", "response"}] for the session's probes (empty if none).
        `memory_user_id` is the namespace reads come from — the harness passes a neighbor's
        id under scramble_namespaces. `oracle_facts` is only supplied for the oracle anchor.
        """
        state: AgentState = {
            "user_id": user_id,
            "memory_user_id": memory_user_id or user_id,
            "session_index": session.index,
            "now": float(session.index) if now is None else now,
            "turns": list(session.turns),
            "probes": list(session.probes),
            "oracle_facts": oracle_facts or {},
            "contexts": {},
            "responses": [],
            "candidates": [],
            "written": 0,
        }
        out = self._graph.invoke(state)
        return out["responses"]


def build_agent(policy: PolicyConfig, store: MemoryStore | None, agent_model, embedder=None) -> Agent:
    """Compile the LangGraph agent for one policy cell.

    One graph shape for every cell — nodes consult `policy` internally rather than the graph
    being rewired per config, so ablation cells differ only in knobs (clean comparison).
    """
    from langgraph.graph import END, START, StateGraph

    if policy.retrieval == "embedding":
        from memprobe.memory.retrieval import EmbeddingRetriever

        retriever = EmbeddingRetriever(store, embedder=embedder)
    else:
        retriever = DirectRetriever(store) if store is not None else None

    def _candidate_records(state: AgentState, probe: Probe) -> list[MemoryRecord]:
        """Fetch, apply the forgetting policy, and cap at top-k for one probe."""
        mem_uid = state["memory_user_id"]
        records: list[MemoryRecord] = []
        if policy.use_episodic and store is not None:
            episodes = [
                r for r in store.all(mem_uid)
                if r.key.startswith(_episodic.EPISODE_KEY_PREFIX + ":")
            ]
            records.extend(episodes)
        if policy.use_semantic and retriever is not None:
            if policy.retrieval == "embedding":
                query = probe_question(probe)  # similarity search sees the natural question
            else:
                query = probe.fact_key         # direct read is a structured key lookup
            # over-fetch so the forgetting policy filters BEFORE the top-k cap
            records.extend(retriever.retrieve(mem_uid, query, k=policy.retrieve_k * 4))
        if policy.forgetting:
            records = apply_forgetting(
                records, now=state["now"], policy=policy.forgetting,
                half_life=policy.half_life_days,
            )
        # Final cap: most-recent k (uniform across backends so k is not a confound), but
        # _render_records re-orders oldest-first for the prompt.
        records = sorted(records, key=lambda r: r.written_at, reverse=True)[: policy.retrieve_k]
        return records

    def retrieve(state: AgentState) -> dict:
        if policy.memory_off or not state["probes"]:
            return {"contexts": {}}
        if policy.oracle:
            lines = [f"my {k} is {v}" for k, v in sorted(state["oracle_facts"].items())]
            return {"contexts": {p.probe_id: list(lines) for p in state["probes"]}}
        contexts = {
            p.probe_id: _render_records(_candidate_records(state, p))
            for p in state["probes"]
        }
        return {"contexts": contexts}

    def respond(state: AgentState) -> dict:
        responses = []
        for probe in state["probes"]:
            context = state["contexts"].get(probe.probe_id, [])
            prompt = "\n".join(
                [SYSTEM_PROMPT]
                + (["What you remember about this user:"] + context if context else [])
                + [probe_question(probe)]
            )
            responses.append({"probe_id": probe.probe_id, "response": agent_model.complete(prompt)})
        return {"responses": responses}

    def extract(state: AgentState) -> dict:
        # Extraction is a model call — only pay for it when a semantic write can follow.
        if policy.memory_off or policy.oracle or not policy.use_semantic:
            return {"candidates": []}
        cands = _semantic.extract_facts(
            state["turns"], agent_model, session_index=state["session_index"]
        )
        return {"candidates": cands}

    def gated_write(state: AgentState) -> dict:
        # Writes ALWAYS target the user's own namespace: scramble redirects reads only, so
        # every user's store gets populated for a neighbor to read (placebo semantics).
        if policy.memory_off or policy.oracle or store is None:
            return {"written": 0}
        written = 0
        if policy.use_semantic and state["candidates"]:
            written = _semantic.apply_write_gate(
                store, state["user_id"], state["candidates"],
                tau=policy.write_gate_tau, now=state["now"],
            )
        if policy.use_episodic:
            summary = _episodic.summarize_session(state["turns"], agent_model)
            _episodic.write_episode(
                store, state["user_id"], state["session_index"], summary, now=state["now"]
            )
        return {"written": written}

    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("respond", respond)
    g.add_node("extract", extract)
    g.add_node("gated_write", gated_write)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "respond")
    g.add_edge("respond", "extract")
    g.add_edge("extract", "gated_write")
    g.add_edge("gated_write", END)

    return Agent(g.compile(), policy, store, agent_model, retriever)
