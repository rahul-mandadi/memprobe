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


def _resp_field(resp, name: str):
    """Read a field off an ollama response, which is a dict in older clients and a
    pydantic ChatResponse in newer ones — support both without pinning."""
    if isinstance(resp, dict):
        return resp.get(name)
    return getattr(resp, name, None)


@dataclass
class OllamaModel:
    """Local Ollama backend (default for real runs; $0, offline).

    The client library is imported lazily so the hermetic core never needs the [local]
    extra installed; a real call without it fails with an actionable message. Token usage
    prefers the server's real counts (prompt_eval_count/eval_count) and falls back to
    word-count approximations so cost plumbing never silently reads zero.
    """

    name: str
    _usage: Usage = field(default_factory=Usage)

    def complete(self, prompt: str) -> str:
        try:
            import ollama
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise RuntimeError(
                "OllamaModel needs the [local] extra and a running Ollama daemon: "
                "pip install -e '.[local]' && ollama serve"
            ) from e
        resp = ollama.chat(model=self.name, messages=[{"role": "user", "content": prompt}])
        message = _resp_field(resp, "message")
        text = message["content"] if isinstance(message, dict) else message.content
        self._usage.input_tokens += int(_resp_field(resp, "prompt_eval_count") or len(prompt.split()))
        self._usage.output_tokens += int(_resp_field(resp, "eval_count") or len(text.split()))
        return text

    @property
    def usage(self) -> Usage:
        return self._usage


@dataclass
class AnthropicModel:
    """Opt-in cloud backend for a stronger judge (Haiku 4.5, ~$1/$5 per Mtok).

    Lazy import + real token usage from the API response. Requires the [api] extra and
    ANTHROPIC_API_KEY; the hermetic core never touches this path. With `bedrock=True` the same
    model runs through AWS Bedrock on the ambient AWS credentials (`[bedrock]` extra), and
    `name` is a Bedrock model or inference-profile id.
    """

    name: str = "claude-haiku-4-5"
    bedrock: bool = False
    region: str = "us-east-1"
    _usage: Usage = field(default_factory=Usage)
    _client: object = field(default=None, repr=False)

    def complete(self, prompt: str) -> str:
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "AnthropicModel needs the [api] extra: pip install -e '.[api]'"
                ) from e
            self._client = (anthropic.AnthropicBedrock(aws_region=self.region)
                            if self.bedrock else anthropic.Anthropic())
        resp = self._client.messages.create(
            model=self.name, max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        self._usage.input_tokens += resp.usage.input_tokens
        self._usage.output_tokens += resp.usage.output_tokens
        return "".join(block.text for block in resp.content if getattr(block, "text", None))

    @property
    def usage(self) -> Usage:
        return self._usage


def build_model(spec: str) -> ChatModel:
    """Factory: 'stub' | 'ollama:<name>' | 'anthropic:<name>' | 'bedrock:<model or profile id>'."""
    if spec == "stub":
        return StubModel()
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        return OllamaModel(name=name or "qwen3")
    if provider == "anthropic":
        return AnthropicModel(name=name or "claude-haiku-4-5")
    if provider == "bedrock":
        return AnthropicModel(name=name or "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                              bedrock=True)
    raise ValueError(f"unknown model spec: {spec!r}")
