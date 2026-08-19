from __future__ import annotations

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from verifiers.v1.acp import ACPConfig, ACPHarness, ACPHarnessSession


def _runner_operations() -> set[str]:
    source = (Path(__file__).parents[2] / "verifiers/v1/acp/runner.py").read_text()
    tree = ast.parse(source)
    operations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        names = {item.id for item in ast.walk(node.left) if isinstance(item, ast.Name)}
        if "operation" not in names:
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                operations.add(comparator.value)
    return operations


def _session() -> ACPHarnessSession:
    session = object.__new__(ACPHarnessSession)
    session.config = ACPConfig(
        env={
            "SECRET": "not serialized by snapshot",
            "CODEX_HOME": "/tmp/vf-codex-home-parent",
        },
        command=["node", "agent.js"],
        prompt="hello",
        system_prompt="system",
        session_meta={"_meta": {"mode": "test"}},
        additional_directories=["/tmp/extra"],
        allow_empty_tool_reply=True,
    )
    session.mcp_urls = {"tools": "http://tool/v1"}
    session.harness = SimpleNamespace(config=SimpleNamespace(id="codex"))
    session.trace = SimpleNamespace(id="trace-1")
    session.requests = []

    async def request(self, packet):
        self.requests.append(packet)
        if packet["operation"] == "snapshot":
            return {
                "ok": True,
                "snapshot": {
                    "session_id": "sess-parent",
                    "cwd": "/work",
                    "additional_directories": ["/work/extra"],
                    "capabilities": {
                        "load_session": True,
                        "session": {"resume": True, "fork": False},
                    },
                },
            }
        return {
            "ok": True,
            "snapshot": {
                "session_id": "sess-child",
                "cwd": packet["config"]["cwd"],
                "capabilities": {},
            },
        }

    session._request = MethodType(request, session)
    return session


def test_runner_exposes_resume_adopt_control_operations():
    assert {
        "prompt",
        "snapshot",
        "resume_session",
        "load_session",
        "fork_session",
        "shutdown",
    } <= _runner_operations()


@pytest.mark.asyncio
async def test_harness_session_snapshot_is_json_safe_and_omits_env():
    session = _session()

    snapshot = await session.snapshot()

    assert snapshot["schema_version"] == "verifiers.acp_session.v1"
    assert snapshot["harness_id"] == "codex"
    assert snapshot["trace_id"] == "trace-1"
    assert snapshot["runner"]["session_id"] == "sess-parent"
    assert snapshot["config"]["command"] == ["node", "agent.js"]
    assert snapshot["config"]["mcp_urls"] == {"tools": "http://tool/v1"}
    assert snapshot["config"]["env"] == {"CODEX_HOME": "/tmp/vf-codex-home-parent"}
    assert "SECRET" not in snapshot["config"]["env"]


@pytest.mark.asyncio
async def test_resume_from_snapshot_prefers_resume_without_fork_capability():
    session = _session()

    await session.resume_from_snapshot(
        {
            "runner": {
                "session_id": "sess-parent",
                "cwd": "/work",
                "additional_directories": ["/work/extra"],
                "capabilities": {
                    "load_session": True,
                    "session": {"resume": True, "fork": False},
                },
            }
        }
    )

    packet = session.requests[-1]
    assert packet["operation"] == "resume_session"
    assert packet["session_id"] == "sess-parent"
    assert packet["config"]["cwd"] == "/work"
    assert packet["config"]["mcp_urls"] == {"tools": "http://tool/v1"}
    assert packet["config"]["session_meta"] == {"_meta": {"mode": "test"}}


@pytest.mark.asyncio
async def test_resume_from_snapshot_uses_fork_when_capability_advertised():
    session = _session()

    await session.resume_from_snapshot(
        {
            "runner": {
                "session_id": "sess-parent",
                "cwd": "/work",
                "capabilities": {
                    "load_session": True,
                    "session": {"resume": True, "fork": True},
                },
            }
        }
    )

    assert session.requests[-1]["operation"] == "fork_session"


class FakeRuntime:
    supports_live_processes = True


class FakeHarness(ACPHarness):
    def __init__(self):
        super().__init__(SimpleNamespace(id="codex"))

    async def prepare_acp(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data):
        return ACPConfig(
            env={"SECRET": "child-secret", "CODEX_HOME": "/tmp/child-home"},
            command=["node", "agent.js"],
            prompt="",
        )


@pytest.mark.asyncio
async def test_adopt_or_resume_reuses_safe_snapshot_env_without_secret(monkeypatch):
    captured = {}

    async def resume_from_snapshot(self, snapshot, *, operation="auto"):
        captured["env"] = dict(self.config.env)
        captured["operation"] = operation
        return {}

    monkeypatch.setattr(ACPHarnessSession, "resume_from_snapshot", resume_from_snapshot)
    harness = FakeHarness()

    await harness.adopt_or_resume(
        ctx="ctx",
        trace=SimpleNamespace(id="child"),
        runtime=FakeRuntime(),
        endpoint="http://model/v1",
        secret="secret",
        mcp_urls={},
        data=None,
        snapshot={
            "runner": {"session_id": "sess-1"},
            "config": {
                "env": {
                    "CODEX_HOME": "/tmp/parent-home",
                    "SECRET": "must-not-override",
                }
            },
        },
        operation="resume_session",
    )

    assert captured["operation"] == "resume_session"
    assert captured["env"]["CODEX_HOME"] == "/tmp/parent-home"
    assert captured["env"]["SECRET"] == "child-secret"
