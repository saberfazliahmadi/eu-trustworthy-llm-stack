"""Offline contract tests for the OpenAI-compatible adapters.

All HTTP traffic is intercepted by `httpx.MockTransport`; no API key, network or model is needed.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

pytest.importorskip("openai")

from trust_llm.providers.base import GenerationRequest  # noqa: E402
from trust_llm.providers.live_provider import (  # noqa: E402
    OpenAICompatibleEmbeddings,
    OpenAICompatibleProvider,
)

BASE_URL = "https://llm.example.test/v1"
API_KEY = "sk-test-not-a-real-key"


def _chat_completion(content: str | None, finish_reason: str = "stop", usage: bool = True) -> dict:
    body = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "test-model-2026",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 21, "completion_tokens": 7, "total_tokens": 28}
    return body


class Recorder:
    """MockTransport handler that stores every request and replies with a canned JSON body."""

    def __init__(self, body: dict, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    @property
    def last_json(self) -> dict:
        return json.loads(self.requests[-1].content)


def _provider(recorder: Recorder, **kwargs) -> OpenAICompatibleProvider:
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    return OpenAICompatibleProvider(
        model="test-model", api_key=API_KEY, base_url=BASE_URL, http_client=client, **kwargs
    )


def test_deterministic_request_payload_and_endpoint():
    rec = Recorder(_chat_completion("Die Probezeit dauert 6 Monate."))
    provider = _provider(rec)
    provider.generate(
        GenerationRequest(
            prompt="Wie lange dauert die Probezeit?",
            system="You are an HR policy assistant.",
            max_tokens=64,
            temperature=0.7,  # ignored: do_sample=False forces greedy decoding
            top_p=0.5,
            seed=42,
            task="rag",
            metadata={"internal": "must not leak to the API"},
        )
    )
    req = rec.requests[-1]
    assert req.method == "POST"
    assert str(req.url) == f"{BASE_URL}/chat/completions"
    payload = rec.last_json
    assert payload["model"] == "test-model"
    assert payload["messages"] == [
        {"role": "system", "content": "You are an HR policy assistant."},
        {"role": "user", "content": "Wie lange dauert die Probezeit?"},
    ]
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.0
    assert payload["top_p"] == 1.0
    assert payload["seed"] == 42
    # task / metadata are mock-simulator hints and must never be sent to a live endpoint
    assert "task" not in payload and "metadata" not in payload


def test_sampling_payload_keeps_temperature_and_top_p():
    rec = Recorder(_chat_completion("ok"))
    _provider(rec).generate(GenerationRequest(prompt="hi", do_sample=True, temperature=1.5, top_p=0.9, seed=7))
    payload = rec.last_json
    assert payload["temperature"] == 1.5
    assert payload["top_p"] == 0.9
    assert payload["seed"] == 7
    assert payload["messages"] == [{"role": "user", "content": "hi"}]  # no system message when unset


def test_auth_header_and_no_key_leak_into_body():
    rec = Recorder(_chat_completion("ok"))
    _provider(rec).generate(GenerationRequest(prompt="hi"))
    req = rec.requests[-1]
    assert req.headers["authorization"] == f"Bearer {API_KEY}"
    assert req.headers["content-type"].startswith("application/json")
    assert API_KEY not in req.content.decode()


def test_response_parsing_uses_server_usage():
    rec = Recorder(_chat_completion("Die Probezeit dauert 6 Monate.", finish_reason="length"))
    result = _provider(rec).generate(GenerationRequest(prompt="Probezeit?"))
    assert result.text == "Die Probezeit dauert 6 Monate."
    assert result.model == "test-model-2026"
    assert result.prompt_tokens == 21
    assert result.completion_tokens == 7
    assert result.finish_reason == "length"


def test_response_parsing_falls_back_without_usage_or_content():
    rec = Recorder(_chat_completion(None, usage=False))
    result = _provider(rec).generate(GenerationRequest(prompt="one two three"))
    assert result.text == ""
    assert result.prompt_tokens > 0  # local token estimate replaces missing server usage
    assert result.completion_tokens >= 0
    assert result.finish_reason == "stop"


def test_missing_api_key_is_rejected_before_any_request():
    rec = Recorder(_chat_completion("ok"))
    client = httpx.Client(transport=httpx.MockTransport(rec))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAICompatibleProvider(model="m", api_key=None, base_url=BASE_URL, http_client=client)
    assert rec.requests == []


def test_http_error_is_raised_not_swallowed():
    import openai

    rec = Recorder({"error": {"message": "invalid key", "type": "invalid_request_error"}}, status=401)
    with pytest.raises(openai.AuthenticationError):
        _provider(rec).generate(GenerationRequest(prompt="hi"))


def test_embeddings_payload_and_l2_normalisation():
    rec = Recorder(
        {
            "object": "list",
            "model": "embed-test",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [3.0, 4.0, 0.0]},
                {"object": "embedding", "index": 1, "embedding": [0.0, 0.0, 2.0]},
            ],
            "usage": {"prompt_tokens": 4, "total_tokens": 4},
        }
    )
    client = httpx.Client(transport=httpx.MockTransport(rec))
    emb = OpenAICompatibleEmbeddings(model="embed-test", api_key=API_KEY, base_url=BASE_URL, http_client=client)
    vecs = emb.embed(["Probezeit", "Urlaub"])
    assert str(rec.requests[-1].url) == f"{BASE_URL}/embeddings"
    payload = rec.last_json
    assert payload["model"] == "embed-test"
    assert payload["input"] == ["Probezeit", "Urlaub"]
    assert rec.requests[-1].headers["authorization"] == f"Bearer {API_KEY}"
    assert vecs.dtype == np.float32 and vecs.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(vecs[0], [0.6, 0.8, 0.0], rtol=1e-6)
    assert emb.dim == 3
