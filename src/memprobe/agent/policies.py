"""Translate a config.MemConfig (one matrix column) into a graph.PolicyConfig.

Fully implemented — pure mapping, no model. Centralizing this keeps the anchor semantics
(memory_off / oracle / placebo) in exactly one place so the harness and tests agree on what
each anchor means.
"""

from __future__ import annotations

from memprobe.agent.graph import PolicyConfig
from memprobe.config import Anchor, MemConfig


def resolve(cfg: MemConfig) -> PolicyConfig:
    mem = cfg.memory
    mem_list = mem if isinstance(mem, list) else [mem]

    return PolicyConfig(
        use_episodic="episodic" in mem_list,
        use_semantic="semantic" in mem_list,
        retrieval=cfg.retrieval,
        write_gate_tau=cfg.write_gate_tau if cfg.write_gate_tau is not None else 0.7,
        forgetting=cfg.forgetting,
        half_life_days=cfg.half_life_days if cfg.half_life_days is not None else 30.0,
        oracle=(cfg.anchor == Anchor.UPPER_BOUND) or (mem == "oracle"),
        memory_off=(cfg.anchor == Anchor.CALIBRATION) or (mem == "none"),
        scramble_namespaces=cfg.scramble_namespaces,
    )
