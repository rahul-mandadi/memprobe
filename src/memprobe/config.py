"""Typed run configuration (parsed from config/default.yaml).

Fully implemented — plain typed loading, no model dependency, so the harness and tests share
one schema. The `anchor` field is what makes eval/anchors.py able to find and validate the
three mandatory anchor configs (NOTES.md ADR-0006).
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Anchor(str, Enum):
    CALIBRATION = "calibration"    # memory_off: must score ~= chance
    UPPER_BOUND = "upper_bound"    # oracle: ground-truth facts injected directly
    PLACEBO = "placebo"            # shuffled: must not beat memory_off


class ScenarioParams(BaseModel):
    n_users: int = 20
    n_sessions: int = 8
    turns_per_session: int = 6
    contradiction_rate: float = 0.25
    distractor_density: float = 0.3
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])


class ModelSpec(BaseModel):
    # Format "provider:name", e.g. "ollama:qwen3", "anthropic:claude-haiku-4-5",
    # "local:all-MiniLM-L6-v2", or "stub" (deterministic, used by the test suite).
    agent: str = "stub"
    judge: str = "stub"
    embeddings: str = "local:all-MiniLM-L6-v2"


class MemConfig(BaseModel):
    """One column of the ablation matrix."""

    name: str
    anchor: Anchor | None = None
    memory: Any = "none"                 # "none" | "oracle" | "episodic" | "semantic" | list
    retrieval: str = "direct"            # "direct" | "embedding"
    write_gate_tau: float | None = None  # ADR-0004 ablation axis
    forgetting: str | None = None        # None | "recency_decay"
    half_life_days: float | None = None
    scramble_namespaces: bool = False    # placebo anchor: serve another user's memory


class ReportSpec(BaseModel):
    confidence: float = 0.95
    out_dir: str = "report/out"


class RunConfig(BaseModel):
    scenarios: ScenarioParams = Field(default_factory=ScenarioParams)
    models: ModelSpec = Field(default_factory=ModelSpec)
    configs: list[MemConfig] = Field(default_factory=list)
    report: ReportSpec = Field(default_factory=ReportSpec)

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)

    def anchor_configs(self) -> list[MemConfig]:
        return [c for c in self.configs if c.anchor is not None]

    def study_configs(self) -> list[MemConfig]:
        return [c for c in self.configs if c.anchor is None]
