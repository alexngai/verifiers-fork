"""Public Agent Client Protocol support for harness programs."""

import asyncio
import contextlib
import json
from abc import abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeVar

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.errors import HarnessError
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.runtimes import ProgramResult, Runtime, RuntimeProcess
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "runner.py").read_text()
MAX_PACKET_BYTES = 128 * 1024 * 1024

__all__ = ["ACPConfig", "ACPHarness"]

ConfigT = TypeVar("ConfigT", bound=HarnessConfig)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
JsonObject: TypeAlias = dict[str, JsonValue]
SNAPSHOT_ENV_KEYS = frozenset({"CODEX_HOME", "CLAUDE_CONFIG_DIR"})


@dataclass
class ACPConfig:
    """One harness's ACP process and initial prompt."""

    env: dict[str, str]
    command: list[str]
    prompt: str | Messages | None
    mcp_urls: dict[str, str] | None = None
    cwd: str | None = None
    additional_directories: list[str] | None = None
    system_prompt: str | None = None
    session_meta: JsonObject | None = None
    allow_empty_tool_reply: bool = False


def _snapshot_env(env: dict[str, str]) -> dict[str, str]:
    """Non-secret env values needed to keep adapter state dirs stable across forks."""
    return {key: value for key, value in env.items() if key in SNAPSHOT_ENV_KEYS}


class ACPHarness(Harness[ConfigT]):
    """Harness backed by one live ACP process and native session per rollout."""

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**self.config.resolved_env, "UV_FROZEN": "false"}
        )

    @abstractmethod
    async def prepare_acp(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ACPConfig:
        pass

    async def session(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> HarnessSession:
        if not runtime.supports_live_processes:
            raise HarnessError(
                f"harness {self.config.id!r} requires a runtime with live process support"
            )
        config = await self.prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        return ACPHarnessSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls if config.mcp_urls is None else config.mcp_urls,
            data,
            config,
        )

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        raise HarnessError(
            f"harness {self.config.id!r} requires a rollout-scoped session"
        )

    async def adopt_or_resume(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        snapshot: JsonObject,
        *,
        operation: str = "auto",
    ) -> "ACPHarnessSession":
        """Create a rollout-scoped ACP session from a prior session snapshot.

        The runtime is expected to already represent the child sandbox/VM state.
        This method reconstructs the host-side ACP runner and asks the adapter to
        resume, load, or fork the native session id captured at the branch point.
        """
        if not runtime.supports_live_processes:
            raise HarnessError(
                f"harness {self.config.id!r} requires a runtime with live process support"
            )
        config = await self.prepare_acp(
            ctx, trace, runtime, endpoint, secret, mcp_urls, data
        )
        snapshot_config = snapshot.get("config")
        if isinstance(snapshot_config, dict):
            env = snapshot_config.get("env")
            if isinstance(env, dict):
                config.env.update(
                    {
                        str(key): str(value)
                        for key, value in env.items()
                        if key in SNAPSHOT_ENV_KEYS
                    }
                )
        session = ACPHarnessSession(
            self,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls if config.mcp_urls is None else config.mcp_urls,
            data,
            config,
        )
        await session.resume_from_snapshot(snapshot, operation=operation)
        return session


def _packet(value: JsonObject) -> bytes:
    data = json.dumps(value, ensure_ascii=False).encode()
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError(f"ACP session packet is too large: {len(data)} bytes")
    return len(data).to_bytes(8, "big") + data


def _require_model_turn(trace: Trace, calls_before: int, result: ProgramResult) -> None:
    if (
        result.exit_code
        or trace.stop_condition is not None
        or any(call.node is not None for call in trace.calls[calls_before:])
    ):
        return
    detail = (result.stderr or result.stdout).strip()[-500:] or "<no output>"
    raise RuntimeError("ACP agent completed without committing a model turn: " + detail)


class _PacketReader:
    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source.__aiter__()
        self._buffer = bytearray()

    async def _readexactly(self, size: int) -> bytes:
        while len(self._buffer) < size:
            try:
                self._buffer.extend(await anext(self._source))
            except StopAsyncIteration as e:
                raise EOFError("ACP process closed its stdout") from e
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data

    async def read(self) -> JsonObject:
        size = int.from_bytes(await self._readexactly(8), "big")
        if size > MAX_PACKET_BYTES:
            raise ValueError(f"ACP session packet is too large: {size} bytes")
        return json.loads((await self._readexactly(size)).decode())


class ACPHarnessSession(HarnessSession):
    """A live ACP process, connection, and native session for one rollout."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        config: ACPConfig,
    ) -> None:
        super().__init__(harness, ctx, trace, runtime, endpoint, secret, mcp_urls, data)
        self.config = config
        self._process: RuntimeProcess | None = None
        self._reader: _PacketReader | None = None
        self._stderr_tail = bytearray()
        self._stderr_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()

    def _prompt_config(self, messages: Messages | None) -> JsonObject:
        prompt = self.config.prompt if messages is None else messages
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        if not isinstance(prompt, str) and (
            not prompt or any(message.role != "user" for message in prompt)
        ):
            raise ValueError("an ACP turn must contain user messages only")
        user_contents = (
            [prompt]
            if isinstance(prompt, str)
            else [
                message.model_dump(mode="json", include={"content"})["content"]
                for message in prompt
            ]
        )
        return self._control_config(user_contents=user_contents)

    def _control_config(
        self,
        *,
        user_contents: list[JsonValue] | None = None,
        runner_snapshot: JsonObject | None = None,
    ) -> JsonObject:
        runner_snapshot = runner_snapshot or {}
        return {
            "command": self.config.command,
            "user_contents": user_contents or [],
            "mcp_urls": self.mcp_urls,
            "cwd": runner_snapshot.get("cwd") or self.config.cwd,
            "additional_directories": (
                runner_snapshot.get("additional_directories")
                or self.config.additional_directories
            ),
            "system_prompt": self.config.system_prompt or "",
            "session_meta": self.config.session_meta or {},
            "allow_empty_tool_reply": self.config.allow_empty_tool_reply,
        }

    @staticmethod
    def _runner_snapshot(snapshot: JsonObject) -> JsonObject:
        runner = snapshot.get("runner", snapshot)
        if not isinstance(runner, dict):
            raise TypeError("ACP session snapshot must contain a runner object")
        return runner

    @staticmethod
    def _choose_resume_operation(runner_snapshot: JsonObject, operation: str) -> str:
        if operation != "auto":
            if operation not in ("resume_session", "load_session", "fork_session"):
                raise ValueError(f"unknown ACP resume operation: {operation!r}")
            return operation
        capabilities = runner_snapshot.get("capabilities")
        if not isinstance(capabilities, dict):
            return "resume_session"
        session = capabilities.get("session")
        if isinstance(session, dict) and session.get("fork"):
            return "fork_session"
        if isinstance(session, dict) and session.get("resume"):
            return "resume_session"
        if capabilities.get("load_session"):
            return "load_session"
        return "resume_session"

    async def _start(self) -> None:
        self._stderr_tail.clear()
        program = await self.runtime.prepare_uv_script(
            ACP_SOURCE,
            {**self.config.env, "UV_FROZEN": "false"},
            activate=False,
        )
        process = await self.runtime.open_process(program, self.config.env)
        self._process = process
        self._reader = _PacketReader(process.stdout)
        self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))

    async def _drain_stderr(self, stream: AsyncIterator[bytes]) -> None:
        async for chunk in stream:
            self._stderr_tail.extend(chunk)
            if len(self._stderr_tail) > 4000:
                del self._stderr_tail[:-4000]

    def _stderr(self) -> str:
        return self._stderr_tail.decode(errors="replace").strip()

    async def _request(self, request: JsonObject) -> JsonObject:
        async with self._lock:
            if self._closed:
                raise HarnessError(
                    f"harness {self.harness.config.id!r} session is already closed"
                )
            if self._process is None:
                await self._start()
            assert self._process is not None
            assert self._reader is not None
            try:
                await self._process.write(_packet(request))
                response = await self._reader.read()
            except BaseException:
                await run_shielded(self._stop(graceful=False))
                raise
        if not response.get("ok"):
            detail = response.get("error") or "ACP session request failed"
            if stderr := self._stderr():
                detail = f"{detail}\n\nACP process stderr:\n{stderr}"
            raise RuntimeError(detail)
        return response

    async def snapshot(self) -> JsonObject:
        response = await self._request({"operation": "snapshot"})
        runner = response.get("snapshot")
        if not isinstance(runner, dict):
            raise TypeError("ACP snapshot response must contain a snapshot object")
        return {
            "schema_version": "verifiers.acp_session.v1",
            "harness_id": self.harness.config.id,
            "trace_id": self.trace.id,
            "runner": runner,
            "config": {
                "command": self.config.command,
                "env": _snapshot_env(self.config.env),
                "mcp_urls": self.mcp_urls,
                "cwd": runner.get("cwd") or self.config.cwd,
                "additional_directories": (
                    runner.get("additional_directories")
                    or self.config.additional_directories
                ),
                "system_prompt": self.config.system_prompt or "",
                "session_meta": self.config.session_meta or {},
                "allow_empty_tool_reply": self.config.allow_empty_tool_reply,
            },
        }

    async def resume_from_snapshot(
        self,
        snapshot: JsonObject,
        *,
        operation: str = "auto",
    ) -> JsonObject:
        runner = self._runner_snapshot(snapshot)
        session_id = runner.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("ACP session snapshot is missing session_id")
        chosen = self._choose_resume_operation(runner, operation)
        response = await self._request(
            {
                "operation": chosen,
                "session_id": session_id,
                "config": self._control_config(runner_snapshot=runner),
            }
        )
        resumed = response.get("snapshot")
        if not isinstance(resumed, dict):
            raise TypeError("ACP resume response must contain a snapshot object")
        return resumed

    async def _run(self, messages: Messages | None) -> ProgramResult:
        config = self._prompt_config(messages)
        calls_before = len(self.trace.calls)
        response = await self._request({"operation": "prompt", "config": config})
        reply = response.get("reply", "")
        if not isinstance(reply, str):
            raise TypeError("ACP session reply must be a string")
        result = ProgramResult(exit_code=0, stdout=reply, stderr="")
        _require_model_turn(self.trace, calls_before, result)
        return result

    async def _stop(self, *, graceful: bool) -> None:
        process, self._process = self._process, None
        reader, self._reader = self._reader, None
        stderr_task, self._stderr_task = self._stderr_task, None
        if process is None:
            return
        try:
            if graceful and reader is not None:
                with contextlib.suppress(BaseException):
                    await process.write(_packet({"operation": "shutdown"}))
                    await asyncio.wait_for(reader.read(), timeout=10)
            for timeout, stop in (
                (10 if graceful else 0.1, None),
                (5, process.terminate),
                (5, process.kill),
            ):
                if stop is not None:
                    with contextlib.suppress(BaseException):
                        await stop()
                try:
                    await asyncio.wait_for(process.wait(), timeout)
                    break
                except TimeoutError:
                    continue
        finally:
            if stderr_task is not None:
                if not stderr_task.done():
                    stderr_task.cancel()
                with contextlib.suppress(BaseException):
                    await stderr_task

    async def close(self) -> None:
        if self._closed:
            return
        # Publish closure before waiting for the process lock. A turn that
        # already passed HarnessSession.turn()'s fast check rechecks under the
        # same lock in _run(), so it cannot restart after teardown.
        await super().close()

        async def close_process() -> None:
            async with self._lock:
                await self._stop(graceful=True)

        await run_shielded(close_process())
