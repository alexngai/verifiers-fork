"""Runtime for a self-hosted AgentENV (E2B-compatible) sandbox plane, with fork acquisition.

This is the rollout-backend adapter that lets the prime-rl orchestrator drive AgentENV instead of
Modal Sandboxes (docs/49 §AgentENV↔prime-rl). It is a first-class fourth `Runtime`, structurally
mirroring `verifiers/v1/runtimes/modal.py` but creating/execing sandboxes through the async E2B SDK
pointed at `CHORUS_E2B_API_URL` (the loopback :8000 plane the orchestrator co-locates with on the
AgentENV node — docs/44 D44-8: the plane is unauthenticated, so loopback is the perimeter).

WHY a dedicated runtime (not config-only): verifiers' RuntimeConfig is a discriminated union with no
URL/endpoint knob, so AgentENV cannot be reached by pointing an existing runtime at it. Every E2B
call here is already live-validated in chorus (`chorus/harness/e2b_container.py`,
`chorus/harness/agentenv_runtime.py`); this file re-shapes them onto the Runtime ABC and is kept
SELF-CONTAINED (only the `e2b` SDK + verifiers base) so it needs no `chorus` import in the image.

TEMPLATES are registered OUT OF BAND (`chorus.harness.agentenv_provision` -> `aenv pull <image>
--name <_slug(image)>`); this runtime references them by the SAME `_slug` and never builds — the
self-hosted plane rejects a raw image ref, only a registered alias resolves. `_slug` is copied
verbatim from e2b_container.py so the aliases line up.

FORK (docs/44, gated behind CHORUS_SANDBOX_FORK / AgentEnvConfig.fork, default OFF). When on, a
per-template LEADER/FOLLOWER gate builds ONE base sandbox per template, snapshots it, and every
rollout of that template FORKS a ready child from the snapshot in ~0.3s (vs a ~40s+ cold create) —
the throughput lever validated live (fork sustains ≥150 concurrent; warm create 0.2-6s). This is a
THROUGHPUT optimization ONLY. It is NOT the fix for the ublk "leak/runaway" (measured 2026-08-19:
serial create+kill is stable, 8-concurrent fork leaks the same as 8-concurrent cold — fork ≈ cold);
that is a node-config collision fixed by raising `ublks_max` above the aenv device-pool watermark
(scripts/agentenv_node_setup.sh). Fork composes with — and does not depend on — that fix.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
)

logger = logging.getLogger(__name__)

# The e2b SDK format-validates its api_key (a bare "dummy" is rejected); AgentENV only
# presence-checks it, so this placeholder passes both. Copied from e2b_container.py.
_PLACEHOLDER_API_KEY = "e2b_" + "0" * 40


def _slug(value: str, *, prefix: str = "chorus-ares") -> str:
    """Template alias for an image ref. VERBATIM copy of e2b_container.py:_slug — the aliases
    MUST match what agentenv_provision registered, or the create call cannot resolve the template."""
    lowered = "".join(ch if ch.isalnum() else "-" for ch in value.lower()).strip("-")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{lowered[:42].strip('-')}-{digest}"


class AgentEnvConfig(BaseConfig):
    type: Literal["agentenv"] = "agentenv"
    image: str = "python:3.11-slim"
    workdir: str = "/app"
    """Default cwd for `run`. Mirrors ModalConfig — the mini_swe_agent harness manages its own
    per-command cwd, so this only needs to exist; the taskset overrides it per task when needed."""
    api_url: str | None = None
    """AgentENV E2B control-plane URL. None -> CHORUS_E2B_API_URL (the co-located loopback plane)."""
    template: str | None = None
    """Explicit template alias override. None -> _slug(image) (the out-of-band-registered alias)."""
    network_access: bool = True
    cpu: float = 1.0
    memory: float = 2.0
    exec_timeout: float = 0.0
    """Per-`run` command timeout in seconds. 0 = no e2b-level timeout (the agent.timeout rollout
    deadline bounds episodes at a higher level); avoids e2b's short default cutting off long steps."""
    sandbox_timeout: int = 24 * 60 * 60
    """Sandbox lifetime (dead-man backstop). Matches Modal's max-lifetime."""
    fork: bool | None = None
    """Fork acquisition. None -> CHORUS_SANDBOX_FORK env (default OFF). When on, one base per template
    is built+snapshotted and every rollout forks a child (~0.3s). Throughput only — see module docs."""


class AgentEnvRuntimeInfo(AgentEnvConfig, BaseRuntimeInfo):
    pass


class AgentEnvRuntime(Runtime):
    is_local: ClassVar[bool] = False
    _endpoint_logged: ClassVar[bool] = False
    # Create-rate gate (measured 2026-08-17). A synchronized initial-batch fill fires ~max_inflight
    # sandbox creates at once; their concurrent SETUP (uv sync + file I/O) saturates the aenv
    # file-I/O path (~24 concurrent) and freezes the whole batch (this was the AgentENV stall).
    # Throttle create ADMISSION so setups stagger: <=1 create per _min_create_interval_s (default
    # 0.5s = 120/min, the measured sustainable arrival rate; 0 disables). Class-level -> shared
    # across every rollout in an env-server worker process. NB: with an N-worker EnvServerPool the
    # aggregate rate is N x this; set the interval / worker count accordingly.
    _create_gate: ClassVar[asyncio.Lock] = asyncio.Lock()
    _last_create_monotonic: ClassVar[float] = 0.0
    _min_create_interval_s: ClassVar[float] = float(
        os.environ.get("CHORUS_AGENTENV_MIN_CREATE_INTERVAL_S", "0.5")
    )
    # Setup-concurrency gate (measured 2026-08-18). The create-RATE gate above staggers admission
    # but does NOT bound how many sandboxes are simultaneously in the I/O-heavy SETUP window
    # (create + verifiers bootstrap-script write + uv sync). A synchronized max_inflight-wide fill
    # put ~64 setups on the node at once and saturated its file-I/O path (~24 concurrent): 37+
    # bootstrap writes failed with an empty WriteTimeout and the survivors never reached a first
    # LLM call (GPUs idle, batch frozen at 0/32). This semaphore is HELD from start() through the
    # first run() (released there or in teardown()), so at most _setup_concurrency sandboxes
    # provision at once while steady-state inflight still climbs to max_inflight. Default 16, safely
    # below the ~24 ceiling; raise via env to trade node-I/O headroom for faster fill.
    _setup_gate: ClassVar[asyncio.Semaphore] = asyncio.Semaphore(
        max(1, int(os.environ.get("CHORUS_AGENTENV_SETUP_CONCURRENCY", "16")))
    )

    # -- fork acquisition (docs/44), shared across every rollout in this worker process. --
    # Default OFF (env CHORUS_SANDBOX_FORK), per-instance overridable via AgentEnvConfig.fork.
    _fork_default: ClassVar[bool] = bool(os.environ.get("CHORUS_SANDBOX_FORK"))
    # template alias -> persistent snapshot id. Insertion order == LRU (move_to_end on hit).
    _snapshots: ClassVar[dict[str, str]] = {}
    # template alias -> "base ready" gate, so exactly ONE base is built even when a whole GRPO
    # group of the same template dispatches at once (the leader builds; the rest await + fork).
    _snapshot_events: ClassVar[dict[str, asyncio.Event]] = {}
    # Snapshots PERSIST (that is what enables cross-rollout reuse), so a large-pool run accumulates
    # them until GC. LRU-cap the live set; evict the oldest beyond the cap via delete_snapshot.
    _snapshot_max: ClassVar[int] = int(os.environ.get("CHORUS_SANDBOX_SNAPSHOT_MAX", "64"))
    # Observability counters (class-wide; surfaced by whatever reads them, mirrors env_service).
    fork_hits: ClassVar[int] = 0        # rollouts served by forking a cached snapshot (~0.3s)
    fork_bases_built: ClassVar[int] = 0  # base sandboxes built + snapshotted (one per template)
    fork_fallbacks: ClassVar[int] = 0   # fork path gave up -> fell back to a fresh cold create
    snapshots_evicted: ClassVar[int] = 0  # snapshots deleted by the LRU cap

    def __init__(self, config: AgentEnvConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = AgentEnvRuntimeInfo(**config.model_dump())
        self._sandbox = None
        self._holds_setup_gate = False
        self._exec_seq = 0
        self._fork_enabled = (
            config.fork if config.fork is not None else type(self)._fork_default
        )

    def _release_setup_gate(self) -> None:
        """Free this rollout's setup-concurrency slot exactly once (first run() or teardown())."""
        if getattr(self, "_holds_setup_gate", False):
            self._holds_setup_gate = False
            try:
                type(self)._setup_gate.release()
            except (ValueError, RuntimeError):  # guard against an accidental over-release
                pass

    def _conn(self) -> dict[str, str]:
        # Endpoint resolution order: explicit config -> env var -> file handoff.
        # The file fallback exists because prime-rl's `rl` launcher spawns the
        # orchestrator as a subprocess that does NOT reliably inherit the driver's
        # CHORUS_E2B_API_URL env, and AgentEnvConfig.api_url is not always threaded
        # through at create time. A container-wide file is robust to both.
        file_url = ""
        try:
            _p = os.environ.get("CHORUS_AGENTENV_URL_FILE", "/tmp/chorus-agentenv-url")
            if os.path.exists(_p):
                with open(_p) as _f:
                    file_url = _f.read().strip()
        except OSError:
            file_url = ""
        base = (
            self.config.api_url or os.environ.get("CHORUS_E2B_API_URL") or file_url or ""
        ).rstrip("/")
        if not type(self)._endpoint_logged:
            type(self)._endpoint_logged = True
            print(
                f"[agentenv-runtime] resolved endpoint {base!r} "
                f"(config={self.config.api_url!r} "
                f"env={os.environ.get('CHORUS_E2B_API_URL')!r} file={file_url!r}) "
                f"fork={self._fork_enabled}",
                flush=True,
            )
        if not base:
            raise SandboxError(
                "AgentEnvRuntime needs a self-hosted AgentENV endpoint: set CHORUS_E2B_API_URL, "
                "AgentEnvConfig.api_url, or write the URL to /tmp/chorus-agentenv-url "
                "(CHORUS_AGENTENV_URL_FILE)"
            )
        return {
            "api_url": base,
            "sandbox_url": base,
            "api_key": os.environ.get("CHORUS_E2B_API_KEY") or _PLACEHOLDER_API_KEY,
        }

    def _template_name(self) -> str:
        return self.config.template or _slug(self.config.image)

    @classmethod
    async def _throttle_create(cls) -> None:
        """Space out sandbox creates so their setups don't all hit the aenv I/O path at once."""
        interval = cls._min_create_interval_s
        if interval <= 0:
            return
        import time
        async with cls._create_gate:
            now = time.monotonic()
            wait = cls._last_create_monotonic + interval - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            cls._last_create_monotonic = now

    async def _cold_create(self, conn: dict[str, str]):
        """Create a fresh sandbox from the registered image template (the ~40s+ cold path)."""
        from e2b import AsyncSandbox

        return await AsyncSandbox.create(
            template=self._template_name(),
            allow_internet_access=self.config.network_access,
            timeout=self.config.sandbox_timeout,
            **conn,
        )

    async def _build_snapshot(self, conn: dict[str, str]) -> str | None:
        """LEADER work: cold-create a base sandbox, capture it as a persistent snapshot, and kill the
        base. Returns the snapshot id (or None on failure -> caller falls back to a fresh create). The
        base is never used for a rollout — create_snapshot PAUSES it — so leader and followers alike
        fork a fresh child from the snapshot. The snapshot captures the ready image filesystem; the
        harness's own per-rollout uv sync still runs on each fork's first run (not baked in here)."""
        base = await self._cold_create(conn)
        try:
            info = await base.create_snapshot()  # pauses the base; returns SnapshotInfo
            snap = getattr(info, "snapshot_id", None) or (info if isinstance(info, str) else None)
            if not snap:
                raise SandboxError(f"create_snapshot returned no usable id: {info!r}")
            return str(snap)
        finally:
            try:
                await asyncio.shield(asyncio.wait_for(base.kill(), timeout=30))
            except Exception:  # noqa: BLE001 — the base is spent; a leaked base is cost, not a bug
                logger.warning("agentenv: failed to kill fork base for %s", self._template_name())

    async def _evict_snapshots(self, conn: dict[str, str]) -> None:
        """Evict oldest snapshots beyond the LRU cap (fork GC). Snapshots survive their source
        sandbox, so a long run must delete its own. Also drops the leadership gate for the evicted
        template so a later rollout re-elects a leader and rebuilds (rather than falling back forever)."""
        from e2b import AsyncSandbox

        cls = type(self)
        while len(cls._snapshots) > cls._snapshot_max:
            tmpl, snap = next(iter(cls._snapshots.items()))  # oldest (insertion/LRU order)
            cls._snapshots.pop(tmpl, None)
            cls._snapshot_events.pop(tmpl, None)
            try:
                await AsyncSandbox.delete_snapshot(snap, **conn)
            except Exception as e:  # noqa: BLE001 — GC is best-effort
                logger.warning("agentenv: snapshot delete failed for %s: %s", snap, e)
            cls.snapshots_evicted += 1

    async def _acquire_via_fork(self, conn: dict[str, str]):
        """Serve a create by forking. Returns a live sandbox, or None to signal the caller to fall
        back to the fresh path. A per-template LEADER/FOLLOWER gate guarantees exactly ONE base build
        even when a whole GRPO group dispatches at once: the first caller builds+snapshots the base,
        the rest await it then fork — so a group of N costs one ~40s base + N-1 ~0.3s forks."""
        from e2b import AsyncSandbox

        cls = type(self)
        tmpl = self._template_name()
        # The section from here through the leader-election set below has NO await, so it is atomic
        # under asyncio — two coroutines cannot both win leadership for the same template.
        snap = cls._snapshots.get(tmpl)
        if snap is not None:
            cls._snapshots[tmpl] = cls._snapshots.pop(tmpl)  # LRU touch (move to most-recent)
        if snap is None:
            ev = cls._snapshot_events.get(tmpl)
            is_leader = ev is None
            if is_leader:
                ev = asyncio.Event()
                cls._snapshot_events[tmpl] = ev
                built = None
                try:
                    built = await self._build_snapshot(conn)
                except Exception as e:  # noqa: BLE001 — base build failed -> fresh fallback
                    logger.warning(
                        "agentenv: fork base build failed for %s (%s: %s); falling back to fresh",
                        tmpl, type(e).__name__, e,
                    )
                finally:
                    # Runs on EVERY exit path incl. cancellation, so followers never hang.
                    if built:
                        cls._snapshots[tmpl] = built
                        cls.fork_bases_built += 1
                    else:
                        cls._snapshot_events.pop(tmpl, None)  # let a later create retry leadership
                    ev.set()
                snap = built
                if snap:
                    await self._evict_snapshots(conn)
            else:
                await ev.wait()
                snap = cls._snapshots.get(tmpl)
        if not snap:
            cls.fork_fallbacks += 1
            return None
        try:
            sandbox = await AsyncSandbox.create(
                template=snap,
                allow_internet_access=self.config.network_access,
                timeout=self.config.sandbox_timeout,
                **conn,
            )
            cls.fork_hits += 1
            return sandbox
        except Exception as e:  # noqa: BLE001 — a bad fork -> fresh fallback (don't wedge the batch)
            logger.warning(
                "agentenv: fork of %s failed (%s: %s); falling back to fresh",
                tmpl, type(e).__name__, e,
            )
            cls.fork_fallbacks += 1
            return None

    async def start(self) -> None:
        # Hold a setup-concurrency slot from create THROUGH the first run() (released there or in
        # teardown()) so at most _setup_concurrency sandboxes are in the I/O-heavy setup window at
        # once — see the _setup_gate note above. Free it here on any failure before the first run().
        await type(self)._setup_gate.acquire()
        self._holds_setup_gate = True
        try:
            await type(self)._throttle_create()  # additionally smooth the create RATE
            try:
                from e2b import AsyncSandbox  # noqa: F401 — import-guard for a clear error
            except ModuleNotFoundError as e:
                raise SandboxError("AgentEnvRuntime requires the `e2b` SDK") from e
            try:
                conn = self._conn()
                # Fork acquisition (throughput): try a warm fork; None -> cold create. When fork is
                # off this is skipped entirely and the path is byte-identical to a plain create.
                if self._fork_enabled:
                    self._sandbox = await self._acquire_via_fork(conn)
                if self._sandbox is None:
                    self._sandbox = await self._cold_create(conn)
                self.info.id = self._sandbox.sandbox_id
                logger.info(
                    "agentenv: sandbox %s up (template=%s, fork=%s)",
                    self.info.id, self._template_name(), self._fork_enabled,
                )
                # Ensure the default workdir exists (mirrors ModalRuntime.start's make_directory).
                try:
                    await self._sandbox.files.make_dir(self.config.workdir)
                except Exception:  # noqa: BLE001 — already exists / not permitted is non-fatal
                    pass
            except SandboxError:
                raise
            except Exception as e:  # provisioning failure is one rollout's problem, not the eval's
                logger.warning("agentenv: provisioning failed for template=%s: %s: %r",
                               self._template_name(), type(e).__name__, e)
                raise SandboxError(
                    f"agentenv sandbox provisioning failed: {type(e).__name__}: {e}"
                ) from e
        except BaseException:  # incl. CancelledError — never leak the slot if we never run()
            self._release_setup_gate()
            raise

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        from e2b import CommandExitException

        # WHY the wrapper: commands.run() streams the exec channel and returns only on stdout/stderr
        # EOF, which needs EVERY process holding fd1/fd2 to release it. So a background child the
        # agent's command spawns (a test daemon, a lingering worker) that INHERITS those fds holds the
        # pipe open and blocks run() until that child dies — a short-lived one just adds latency; a
        # daemon wedges the rollout forever (the AgentENV "5 done, 27 hung" freeze). Measured CPU-only
        # on cloud e2b 2026-08-18: `(sleep 600) & echo` inheriting fd1 -> run() HANGS past 40s;
        # setsid/disown do NOT help; only redirecting the streams does.
        #
        # THE FIX: run the command with its stdout/stderr redirected to files, then cat the files back
        # over the exec channel. Orphaned children inherit the FILE fds (fd1=out, fd2=err), never the
        # exec pipe, so the exec channel EOFs the instant the FOREGROUND command exits regardless of
        # what it left running; `cat` of a regular file returns at end-of-file without waiting for a
        # concurrent writer. Pure shell + the exec channel only (no files API) -> behaves identically
        # on stock cloud e2b and the self-hosted aenv plane. The trailing `exit` preserves the inner
        # command's exit code, so a non-zero exit still surfaces as CommandExitException as before.
        self._exec_seq += 1
        base = f"/tmp/.chorus-exec.{self._exec_seq}"
        out_q = shlex.quote(f"{base}.out")
        err_q = shlex.quote(f"{base}.err")
        # A subshell `( ... )` (not a brace group) wraps the command: it needs no trailing `;`
        # before the close paren and stays valid for ANY joined command, incl. one that ends in a
        # background `&` (a brace group would be a `& ; }` syntax error). Validated end-to-end on
        # cloud e2b 2026-08-18 (W6): returns in 0.2s with the daemon still running, output captured.
        wrapped = (
            f"( {shlex.join(argv)} ) >{out_q} 2>{err_q} </dev/null; "
            f"__chorus_rc=$?; cat {out_q}; cat {err_q} >&2; exit $__chorus_rc"
        )
        try:
            result = await self._sandbox.commands.run(
                wrapped,
                # `env or {}` (not process_env): matches the proven shipping behavior and takes no
                # dependency on a base method that may be absent in an older pinned verifiers when
                # this file is injected into it. self.env is empty on the mini_swe_agent path, so
                # this is byte-identical to process_env(env) there anyway.
                envs=env or {},
                user="root",
                cwd=self.config.workdir or None,
                timeout=self.config.exec_timeout,
            )
            return ProgramResult(
                exit_code=int(result.exit_code),
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        except CommandExitException as e:
            # Command RAN with a non-zero exit — a normal result, not a sandbox failure. The trailing
            # cats still ran, so its captured stdout/stderr are on e.stdout/e.stderr as before.
            return ProgramResult(
                exit_code=int(e.exit_code),
                stdout=e.stdout or "",
                stderr=e.stderr or "",
            )
        except Exception as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"agentenv exec failed: {e}") from e
        finally:
            # The first exec means setup (create + bootstrap write + uv sync) is done and the box is
            # warm; free the setup slot so the next queued rollout can provision. No-op after the 1st.
            self._release_setup_gate()

    async def _read(self, path: str) -> bytes:
        try:
            return await self._sandbox.files.read(path, format="bytes", user="root")
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        try:
            await self._sandbox.files.write(path, data, user="root", use_octet_stream=True)
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    async def teardown(self) -> None:
        # Robustly reap the microVM. A leaked sandbox pins ublk/overlaybd devices AND CPU, and leaked
        # boxes accumulate over a long run. The OLD one-shot `await kill()` leaked two ways: (1) kill()
        # can FAIL under load (was swallowed with no retry), (2) a rollout being torn down is often
        # itself being CANCELLED (dispatcher off-policy drop), which cancels the kill mid-flight. Fix
        # both: shield each kill from cancellation so it still runs to completion, and retry on failure.
        # The sandbox_timeout is only the last-resort backstop. NB: forked children are ordinary
        # sandboxes — killed the same way; the persistent SNAPSHOTS they forked from are GC'd
        # separately by the LRU cap (_evict_snapshots), not here.
        self._release_setup_gate()  # a rollout torn down before its first run() still holds a slot
        sandbox = self._sandbox
        if sandbox is None:
            return
        self._sandbox = None  # idempotent — never retry the same handle from a second call
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                await asyncio.shield(asyncio.wait_for(sandbox.kill(), timeout=30))
                return
            except asyncio.CancelledError:
                # Our awaiter is cancelled, but shield keeps the kill running to completion in the
                # background so the box is still reaped. Preserve cancellation for the caller.
                raise
            except Exception as e:  # noqa: BLE001 — provider teardown is best-effort
                last_err = e
                await asyncio.sleep(1.0 * (attempt + 1))
        logger.warning(
            "agentenv: failed to kill sandbox %s after retries: %s", self.info.id, last_err
        )
