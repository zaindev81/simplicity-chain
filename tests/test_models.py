from __future__ import annotations

import json

import httpx
import pytest

from simplicity_chain import ChatMessage, ModelError, OpenAIServerModel, TokenUsage


def ok_response(content: str = "hello", usage: bool = True) -> httpx.Response:
    body = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 7}
    return httpx.Response(200, json=body)


def make_model(handler, **kwargs) -> OpenAIServerModel:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("api_base", "https://api.test/v1")
    kwargs.setdefault("api_key", "test-key")
    kwargs.setdefault("retry_backoff", 0)
    return OpenAIServerModel("test-model", http_client=client, **kwargs)


def test_request_payload_and_response_parsing():
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ok_response("hi there")

    model = make_model(handler)
    messages = [
        ChatMessage(role="system", content="be nice"),
        ChatMessage(role="user", content="hello"),
    ]
    reply = model.generate(messages, stop_sequences=["Observation:"])

    request = captured[0]
    assert str(request.url) == "https://api.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    body = json.loads(request.content)
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]
    assert body["stop"] == ["Observation:"]
    assert body["temperature"] == 0.2

    assert reply.role == "assistant"
    assert reply.content == "hi there"
    assert reply.token_usage == TokenUsage(11, 7)
    assert reply.raw["choices"][0]["message"]["content"] == "hi there"


def test_retries_on_server_error_then_succeeds():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(500, text="boom")
        return ok_response()

    model = make_model(handler, max_retries=1)
    assert model.generate([ChatMessage(role="user", content="hi")]).content == "hello"
    assert attempts["count"] == 2


def test_client_errors_are_not_retried():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        return httpx.Response(401, text="bad key")

    model = make_model(handler)
    with pytest.raises(ModelError, match="HTTP 401"):
        model.generate([ChatMessage(role="user", content="hi")])
    assert attempts["count"] == 1


def test_transport_errors_are_retried_then_reported():
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("connection refused")

    model = make_model(handler, max_retries=1)
    with pytest.raises(ModelError, match="failed"):
        model.generate([ChatMessage(role="user", content="hi")])
    assert attempts["count"] == 2


def test_missing_api_key_raises_at_construction(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ModelError, match="No API key"):
        OpenAIServerModel("test-model", api_base="https://api.test/v1")


def test_empty_choices_raise():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    model = make_model(handler)
    with pytest.raises(ModelError, match="no choices"):
        model.generate([ChatMessage(role="user", content="hi")])


def test_invalid_json_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    model = make_model(handler)
    with pytest.raises(ModelError, match="invalid JSON"):
        model.generate([ChatMessage(role="user", content="hi")])


def test_unexpected_json_payload_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=["not", "an", "object"])

    model = make_model(handler)
    with pytest.raises(ModelError, match="unexpected JSON"):
        model.generate([ChatMessage(role="user", content="hi")])


def test_missing_usage_gives_no_token_usage():
    def handler(request: httpx.Request) -> httpx.Response:
        return ok_response(usage=False)

    model = make_model(handler)
    reply = model.generate([ChatMessage(role="user", content="hi")])
    assert reply.token_usage is None


def test_list_content_is_flattened():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "part one"},
                                {"type": "text", "text": "part two"},
                            ],
                        }
                    }
                ]
            },
        )

    model = make_model(handler)
    assert model.generate([ChatMessage(role="user", content="hi")]).content == "part one\npart two"


def test_api_base_comes_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    captured = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return ok_response()

    client = httpx.Client(transport=httpx.MockTransport(handler))
    model = OpenAIServerModel("test-model", http_client=client)
    model.generate([ChatMessage(role="user", content="hi")])
    assert str(captured[0].url) == "https://env.test/v1/chat/completions"
    assert captured[0].headers["authorization"] == "Bearer env-key"


def test_close_is_safe():
    model = OpenAIServerModel("test-model", api_key="test-key", api_base="https://api.test/v1")
    model.close()
