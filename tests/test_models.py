"""Model factory + the Ollama adapter (exercised against a fake client, no network)."""

import sys
import types

import pytest

from memprobe.agent.models import OllamaModel, StubModel, build_model


def test_factory_dispatch():
    assert isinstance(build_model("stub"), StubModel)
    m = build_model("ollama:qwen3")
    assert isinstance(m, OllamaModel) and m.name == "qwen3"
    assert build_model("ollama:").name == "qwen3"  # default model name
    with pytest.raises(ValueError):
        build_model("mystery:model")


def _fake_ollama(monkeypatch, response):
    fake = types.ModuleType("ollama")
    calls = []

    def chat(model, messages):
        calls.append({"model": model, "messages": messages})
        return response

    fake.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", fake)
    return calls


def test_ollama_uses_server_token_counts(monkeypatch):
    calls = _fake_ollama(monkeypatch, {
        "message": {"content": "you are on pro"},
        "prompt_eval_count": 12,
        "eval_count": 4,
    })
    m = OllamaModel(name="qwen3")
    assert m.complete("what's my plan?") == "you are on pro"
    assert calls[0]["model"] == "qwen3"
    assert m.usage.input_tokens == 12 and m.usage.output_tokens == 4


def test_ollama_falls_back_to_word_counts(monkeypatch):
    _fake_ollama(monkeypatch, {"message": {"content": "two words"}})
    m = OllamaModel(name="qwen3")
    m.complete("a three word prompt")
    assert m.usage.input_tokens == 4 and m.usage.output_tokens == 2


def test_ollama_supports_object_style_responses(monkeypatch):
    message = types.SimpleNamespace(content="object style reply")
    resp = types.SimpleNamespace(message=message, prompt_eval_count=7, eval_count=3)
    _fake_ollama(monkeypatch, resp)
    m = OllamaModel(name="qwen3")
    assert m.complete("hi") == "object style reply"
    assert m.usage.input_tokens == 7 and m.usage.output_tokens == 3
