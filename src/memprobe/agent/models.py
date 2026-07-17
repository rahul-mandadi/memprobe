"""Model adapters. One interface, several backends, and a deterministic stub for tests.

Default real backend is local Ollama ($0, offline); embeddings default to local
sentence-transformers. Anthropic (Haiku 4.5) is opt-in for a stronger judge. The `StubModel`
is deterministic and model-free so the whole harness — and CI — runs with no keys and no
Ollama (NOTES.md ADR-0005: the lab must be exercisable without a model).

`StubModel` implemented; real backends stubbed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


class ChatModel(Protocol):
    def complete(self, prompt: str) -> str: ...
    @property
    def usage(self) -> Usage: ...


class StubModel:
    """Deterministic, model-free backend.

    Its `complete` echoes back the last 'my <key> is <value>' style fact it sees in the prompt,
    which lets the end-to-end pipeline and anchors run in tests without a real model: an oracle
    prompt (facts injected) yields correct answers; a memory-off prompt yields none. It is NOT
    a substitute for a real agent — only a harness driver. Token counts are word-count
    approximations so cost plumbing is exercised.
    """

    def __init__(self) -> None:
        self._usage = Usage()

    def complete(self, prompt: str) -> str:
        import re

        self._usage.input_tokens += len(prompt.split())
        matches = re.findall(r"my ([\w]+) is (?:now )?([\w]+)", prompt.lower())
        # Answer with the LAST stated value per key (recency), mimicking a memory that has the
        # current fact when it's present in context and nothing when it isn't.
        latest: dict[str, str] = {}
        for k, v in matches:
            latest[k] = v
        answer = "; ".join(f"{k}: {v}" for k, v in latest.items()) if latest else "i don't have that on file"
        self._usage.output_tokens += len(answer.split())
        return answer

    @property
    def usage(self) -> Usage:
        return self._usage


@dataclass
class OllamaModel:
    """Local Ollama backend (default for real runs). TODO(milestone-2)."""

    name: str
    _usage: Usage = field(default_factory=Usage)

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("TODO(milestone-2): call ollama.chat; accumulate token usage")

    @property
    def usage(self) -> Usage:
        return self._usage


@dataclass
class AnthropicModel:
    """Opt-in cloud backend for a stronger judge (Haiku 4.5, ~$1/$5 per Mtok). TODO(milestone-4)."""

    name: str = "claude-haiku-4-5"
    _usage: Usage = field(default_factory=Usage)

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("TODO(milestone-4): anthropic messages API; real token usage")

    @property
    def usage(self) -> Usage:
        return self._usage


def build_model(spec: str) -> ChatModel:
    """Factory: 'stub' | 'ollama:<name>' | 'anthropic:<name>'. Implemented for stub."""
    if spec == "stub":
        return StubModel()
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        return OllamaModel(name=name or "qwen3")
    if provider == "anthropic":
        return AnthropicModel(name=name or "claude-haiku-4-5")
    raise ValueError(f"unknown model spec: {spec!r}")
