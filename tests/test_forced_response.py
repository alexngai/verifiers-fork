from __future__ import annotations

from types import SimpleNamespace

import pytest

from verifiers.v1.dialects import ChatDialect
from verifiers.v1.dialects.anthropic import AnthropicDialect
from verifiers.v1.dialects.responses import ResponsesDialect
from verifiers.v1.errors import TaskError
from verifiers.v1.interception.server import _forced_raw_response
from verifiers.v1.session import RolloutSession, request_fingerprint
from verifiers.v1.types import (
    AssistantMessage,
    Request,
    Response,
    ToolCall,
    Usage,
    UserMessage,
)


def test_rollout_session_consumes_one_shot_forced_response() -> None:
    session = RolloutSession(ctx=SimpleNamespace(model="model-a"), trace=object())
    request = Request(messages=[UserMessage(content="next")])
    fingerprint = request_fingerprint(request)

    session.enqueue_forced_response(
        {
            "content": "forced assistant text",
            "prompt_fingerprint": fingerprint,
            "source": "mcts-test",
            "metadata": {"branch_id": "b1"},
        }
    )

    forced = session.consume_forced_response(request)

    assert forced is not None
    assert forced.response.model == "model-a"
    assert forced.response.message.content == "forced assistant text"
    assert forced.prompt_fingerprint == fingerprint
    assert forced.source == "mcts-test"
    assert forced.metadata == {"branch_id": "b1"}
    assert session.consume_forced_response(request) is None


def test_rollout_session_rejects_forced_response_prompt_mismatch() -> None:
    session = RolloutSession(ctx=SimpleNamespace(model="model-a"), trace=object())
    request = Request(messages=[UserMessage(content="expected")])
    other = Request(messages=[UserMessage(content="actual")])
    session.enqueue_forced_response(
        {
            "content": "wrong boundary",
            "prompt_fingerprint": request_fingerprint(request),
        }
    )

    with pytest.raises(TaskError, match="prompt fingerprint mismatch"):
        session.consume_forced_response(other)


def test_forced_raw_response_builds_chat_completion_shape() -> None:
    response = Response(
        id="forced-1",
        created=123,
        model="model-a",
        message=AssistantMessage(content="hello"),
        finish_reason="stop",
        usage=Usage(prompt_tokens=3, completion_tokens=2),
    )

    raw = _forced_raw_response(ChatDialect(), {"model": "ignored"}, response)

    assert raw["id"] == "forced-1"
    assert raw["object"] == "chat.completion"
    assert raw["choices"][0]["message"] == {
        "role": "assistant",
        "content": "hello",
    }
    assert raw["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_forced_raw_response_builds_chat_tool_call_shape() -> None:
    response = Response(
        id="forced-tool",
        created=123,
        model="model-a",
        message=AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="edit_file", arguments='{"path":"a.py"}')
            ],
        ),
        finish_reason="tool_calls",
    )

    raw = _forced_raw_response(ChatDialect(), {"model": "ignored"}, response)

    choice = raw["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "edit_file", "arguments": '{"path":"a.py"}'},
        }
    ]


def test_forced_raw_response_builds_anthropic_tool_call_shape() -> None:
    response = Response(
        id="forced-tool",
        created=123,
        model="claude-test",
        message=AssistantMessage(
            content="",
            tool_calls=[
                ToolCall(id="toolu_1", name="edit_file", arguments='{"path":"a.py"}')
            ],
        ),
        finish_reason="tool_calls",
    )

    raw = _forced_raw_response(AnthropicDialect(), {"model": "ignored"}, response)

    assert raw["type"] == "message"
    assert raw["role"] == "assistant"
    assert raw["stop_reason"] == "tool_use"
    assert raw["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "edit_file",
            "input": {"path": "a.py"},
        }
    ]


def test_forced_raw_response_builds_responses_tool_call_shape() -> None:
    response = Response(
        id="forced-tool",
        created=123,
        model="gpt-test",
        message=AssistantMessage(
            content=None,
            tool_calls=[
                ToolCall(id="call_1", name="edit_file", arguments='{"path":"a.py"}')
            ],
        ),
        finish_reason="tool_calls",
    )

    raw = _forced_raw_response(ResponsesDialect(), {"model": "ignored"}, response)

    assert raw["object"] == "response"
    assert raw["output"] == [
        {
            "type": "function_call",
            "id": "call_1",
            "call_id": "call_1",
            "name": "edit_file",
            "arguments": '{"path":"a.py"}',
            "status": "completed",
        }
    ]


def test_forced_raw_response_preserves_exact_raw_payload() -> None:
    raw_payload = {
        "id": "provider-native",
        "output": [{"type": "reasoning", "encrypted_content": "opaque"}],
    }
    response = Response(
        id="forced-raw",
        created=123,
        model="model-a",
        message=AssistantMessage(content="ignored"),
        finish_reason="stop",
        raw=raw_payload,
    )

    assert _forced_raw_response(ChatDialect(), {"model": "ignored"}, response) == raw_payload
