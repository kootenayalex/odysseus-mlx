"""Cookbook serve lifecycle: kills scheduler-owned serves whose end-of-
window has passed.

Pairs with action_cookbook_serve in builtin_actions.py — that action
stamps the task it launches with `_scheduledStopAtMs`, this loop ticks
every 60s and kills any serve whose stamp is in the past.

Single small module. Delete this file + the registration line in app.py
and the feature stops doing anything; scheduler-launched serves just
stay up until the user kills them manually.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx
from core.constants import internal_api_base
from src.constants import COOKBOOK_STATE_FILE

logger = logging.getLogger(__name__)


def _internal_headers() -> dict:
    from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN
    return {INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}


async def _delete_endpoint_for_task(task: dict) -> None:
    """Drop the auto-registered model endpoint for a scheduled-stop serve.

    Without this, killing the tmux session leaves the endpoint sitting in
    the picker (probe goes offline; chats still try to route there) and
    the user has to delete it by hand in Settings -> Endpoints.
    """
    import re as _re
    payload = task.get("payload") or {}
    cmd = str(payload.get("_cmd") or "")
    remote = task.get("remoteHost") or ""
    # Build host the same way _auto_register_llm_endpoint does so URL match wins.
    if remote:
        host = remote.split("@")[-1] if "@" in remote else remote
    else:
        host = "host.docker.internal"
    port_match = _re.search(r"--port\s+(\d+)", cmd)
    ollama_host_match = _re.search(r"OLLAMA_HOST=[^\s]*?:(\d+)", cmd)
    if port_match:
        port = int(port_match.group(1))
    elif ollama_host_match:
        port = int(ollama_host_match.group(1))
    elif "ollama" in cmd:
        port = 11434
    else:
        port = 8080
    base_url = f"http://{host}:{port}/v1"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                f"{internal_api_base()}/api/model-endpoints",
                headers=_internal_headers(),
            )
            if r.status_code >= 400:
                return
            eps = r.json() if r.content else []
            # Prefer exact URL match; fall back to host:port substring so we
            # still catch the case where 0.0.0.0 vs the registered host
            # representation diverged.
            ep = next((e for e in eps if e.get("base_url") == base_url), None)
            if not ep:
                hostport = f"{host}:{port}"
                ep = next((e for e in eps if hostport in (e.get("base_url") or "")), None)
            if ep:
                await client.delete(
                    f"{internal_api_base()}/api/model-endpoints/{ep['id']}",
                    headers=_internal_headers(),
                )
                logger.info(
                    f"cookbook_serve_lifecycle: deleted endpoint {ep.get('id')} "
                    f"({ep.get('base_url')}) after scheduled stop"
                )
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: endpoint delete failed: {e}")


async def _stop_serve(session_id: str, remote_host: str = "", ssh_port: str = "") -> bool:
    """Kill the tmux session that hosts the serve.

    There's no `/api/model/stop` route — the cookbook UI and the chat
    agent both kill via `/api/shell/exec` running a `tmux kill-session`
    (wrapped in ssh for remote hosts). Mirror that here so the
    lifecycle loop can actually stop scheduler-launched serves at
    window-end. Without this, the action stamped `_scheduledStopAtMs`
    correctly but every kill attempt failed silently (the route
    returned 404 and the result was logged as "failed").
    """
    import shlex
    if remote_host:
        port_flag = f"-p {shlex.quote(str(ssh_port))} " if ssh_port and str(ssh_port) != "22" else ""
        cmd = (
            f"ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no "
            f"{port_flag}{shlex.quote(remote_host)} "
            f"'tmux kill-session -t {shlex.quote(session_id)}'"
        )
    else:
        cmd = f"tmux kill-session -t {shlex.quote(session_id)}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{internal_api_base()}/api/shell/exec",
                json={"command": cmd},
                headers=_internal_headers(),
            )
            if r.status_code >= 400:
                return False
            data = r.json() if r.content else {}
            ec = data.get("exit_code")
            # tmux returns non-zero when the session is already gone
            # ("can't find session: ..."). That's still "stop succeeded"
            # from our POV — the goal is no live session at the end.
            if ec in (None, 0):
                return True
            stderr = (data.get("stderr") or "").lower()
            return "no server" in stderr or "can't find session" in stderr or "session not found" in stderr
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: stop {session_id} failed: {e}")
        return False


async def _delete_endpoint_by_id(endpoint_id: str) -> None:
    """Drop a model endpoint by id (used when evicting a scheduler-owned MLX
    serve, where we already know the auto-registered endpoint id)."""
    if not endpoint_id:
        return
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            await client.delete(
                f"{internal_api_base()}/api/model-endpoints/{endpoint_id}",
                headers=_internal_headers(),
            )
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: endpoint delete {endpoint_id} failed: {e}")


def _kill_local_serve_proc(serve) -> bool:
    """Kill the detached local process backing `serve`, resolved by pid (with a
    live-port fallback). Returns True if a process was signalled.

    Local serves are launched detached, so `tmux kill-session` does NOT reap the
    serve process — it only closes the (already-gone) shell. This is what
    actually frees the multi-GB unified-memory footprint. No-op for remote serves
    (their process lives on the far host; the ssh tmux-kill handles those)."""
    from core.platform_compat import kill_process_tree, pid_alive
    from services import mlx_scheduler as ms

    if serve.remote_host:
        return False
    pid = serve.pid if pid_alive(serve.pid) else ms.local_serve_pid(serve.port)
    if not pid:
        return False
    # Guardrail: never kill an externally-supervised serve (e.g. launchd util).
    for p in ms.discover_local_mlx_procs():
        if p["pid"] == pid and ms.is_protected_cmd(p["cmd"]):
            return False
    kill_process_tree(pid)
    # kill_process_tree sends SIGTERM; rapid-mlx traps it for a graceful
    # shutdown that can hang mid-load, so escalate to SIGKILL if it lingers.
    import os
    import signal
    import time as _t
    for _ in range(15):
        if not pid_alive(pid):
            return True
        _t.sleep(0.2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass
    return True


async def evict_serve(serve, reason: str = "evicted") -> bool:
    """Stop one MLX serve and forget it: kill its tmux session AND its detached
    serve process, delete its auto-registered endpoint, and remove it from the
    scheduler registry.

    `serve` is a services.mlx_scheduler.LoadedServe. This is the single kill
    path shared by budget eviction (in the serve route) and idle-TTL reaping
    (in the loop below), so both stay consistent.
    """
    from services import mlx_scheduler as ms

    ok = await _stop_serve(serve.session_id, serve.remote_host or "", serve.ssh_port or "")
    # Detached local serves outlive their tmux session — kill the process too,
    # else the tmux-kill "succeeds" while the multi-GB serve keeps running (the
    # exact leak this module now guards against).
    if not serve.remote_host:
        try:
            await asyncio.to_thread(_kill_local_serve_proc, serve)
        except Exception as e:
            logger.warning(f"cookbook_serve_lifecycle: proc kill {serve.session_id} failed: {e}")
    if serve.endpoint_id:
        await _delete_endpoint_by_id(serve.endpoint_id)
    else:
        # Fall back to host:port matching when we don't have the endpoint id.
        await _delete_endpoint_for_task({
            "payload": {"_cmd": f"--port {serve.port}"} if serve.port else {},
            "remoteHost": serve.remote_host or "",
        })
    try:
        ms.deregister_serve(serve.session_id)
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: deregister {serve.session_id} failed: {e}")
    logger.info(f"cookbook_serve_lifecycle: evicted MLX serve {serve.session_id} ({reason})")
    return ok


async def _session_alive(session_id: str) -> bool:
    """True if the tmux session still exists locally. Used to reconcile the
    scheduler registry against serves that died outside our control."""
    import shlex
    cmd = f"tmux has-session -t {shlex.quote(session_id)} 2>/dev/null"
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.post(
                f"{internal_api_base()}/api/shell/exec",
                json={"command": cmd},
                headers=_internal_headers(),
            )
            if r.status_code >= 400:
                return True  # don't reap on an inconclusive probe
            data = r.json() if r.content else {}
            return data.get("exit_code") in (None, 0)
    except Exception:
        return True  # never reap a live serve on a transient probe failure


async def _serve_alive(serve) -> bool:
    """Authoritative liveness for a *local* serve: alive if its detached process
    is still running (pid or live-port), even when its tmux session is gone.

    This is the fix for the orphan leak: the old code treated a missing tmux
    session as "dead" and deregistered the serve — dropping the bookkeeping while
    the detached multi-GB process kept running, invisible to reuse and eviction.
    Now the process is the source of truth; the serve stays tracked (and thus
    TTL-reapable, which actually kills it) until the process is really gone.
    Remote serves keep the tmux-session probe (their process is off-box)."""
    from core.platform_compat import pid_alive
    from services import mlx_scheduler as ms

    if serve.remote_host:
        return await _session_alive(serve.session_id)
    if pid_alive(serve.pid):
        return True
    if ms.local_serve_pid(serve.port) is not None:
        return True
    # No live process on pid or port — fall back to the tmux probe so a serve
    # that hasn't recorded a pid/port yet isn't reaped prematurely.
    return await _session_alive(serve.session_id)


# Grace period before the periodic sweep will kill an unregistered serve, so a
# serve that has bound its port but not yet landed in the registry (the brief
# launch→register window) is never reaped mid-startup. The boot reconcile uses
# no grace — at boot nothing of ours is mid-launch.
_ORPHAN_MIN_AGE_S = 150


def _orphan_sweep(known_ports: set, *, min_age_s: int) -> int:
    """Kill live local MLX serve processes that the registry doesn't know about
    and nothing supervises. Returns the number killed.

    This reclaims serves the scheduler lost track of — chiefly detached serves
    that survived an `io.odysseus.server` restart (the registry reconciles to
    empty on boot while the processes live on). Without this, every subsequent
    autoserve spawns a fresh duplicate and none are ever reaped -> RAM leak.

    Protected: externally-supervised serves (launchd rapid-util) and any serve
    whose port is in `known_ports`. `min_age_s` shields just-launched serves."""
    from core.platform_compat import kill_process_tree, pid_alive
    from services import mlx_scheduler as ms

    killed = 0
    for p in ms.discover_local_mlx_procs():
        port, pid, cmd = p["port"], p["pid"], p["cmd"]
        if ms.is_protected_cmd(cmd):
            continue
        if port in known_ports:
            continue
        if p["age_s"] < min_age_s:
            continue
        logger.warning(
            "cookbook_serve_lifecycle: killing orphan MLX serve pid=%s port=%s age=%ss "
            "(not in registry, unsupervised)", pid, port, p["age_s"]
        )
        try:
            kill_process_tree(pid)
            import os
            import signal
            import time as _t
            for _ in range(15):
                if not pid_alive(pid):
                    break
                _t.sleep(0.2)
            if pid_alive(pid):
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except Exception:
                    os.kill(pid, signal.SIGKILL)
            killed += 1
        except Exception as e:
            logger.warning(f"cookbook_serve_lifecycle: orphan kill pid={pid} failed: {e}")
    return killed


def reconcile_orphans_on_boot() -> int:
    """One-shot startup sweep: kill any pre-existing unsupervised MLX serve not
    in the registry. Runs synchronously at startup (before the periodic loop) so
    a server restart doesn't leave last generation's detached serves leaking
    unified memory. Returns the number killed."""
    from services import mlx_scheduler as ms

    try:
        known_ports = {s.port for s in ms.load_registry().values() if s.port}
    except Exception:
        known_ports = set()
    try:
        n = _orphan_sweep(known_ports, min_age_s=0)
        if n:
            logger.info("cookbook_serve_lifecycle: boot orphan sweep killed %s serve(s)", n)
        return n
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: boot orphan sweep failed: {e}")
        return 0


async def _reap_mlx_serves() -> None:
    """TTL idle-unload + registry/process reconciliation for MLX serves.

    Three passes, in order:
      1. Orphan sweep — kill live local serves the registry doesn't track and
         nothing supervises (self-heals the restart-orphan leak even mid-run).
      2. Reconcile — drop registry entries whose process is genuinely gone, so
         budget accounting reflects freed memory.
      3. Idle-TTL — evict serves idle past their TTL (this now also kills the
         detached process, not just the tmux session).

    Not gated on a non-empty registry: an empty registry with live orphans is
    precisely the leak state, and pass 1 must still run.
    """
    from services import mlx_scheduler as ms

    # 1) Orphan sweep (registry-independent).
    try:
        known_ports = {s.port for s in ms.load_registry().values() if s.port}
        await asyncio.to_thread(_orphan_sweep, known_ports, min_age_s=_ORPHAN_MIN_AGE_S)
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: orphan sweep failed: {e}")

    reg = ms.load_registry()
    if not reg:
        return
    loaded = list(reg.values())
    # 2) Reconcile: drop entries whose process is really gone (pid/port/tmux all
    #    dead), so freed memory is reflected in the budget.
    for serve in loaded:
        if not serve.remote_host and not await _serve_alive(serve):
            try:
                ms.deregister_serve(serve.session_id)
            except Exception:
                pass
    # 3) Idle-TTL reap on the reconciled view.
    reg = ms.load_registry()
    for sid in ms.expired(list(reg.values())):
        serve = reg.get(sid)
        if serve:
            await evict_serve(serve, reason="ttl")


async def _tick() -> None:
    state_path = Path(COOKBOOK_STATE_FILE)
    if not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("cookbook_serve_lifecycle: state file unreadable (%s), skipping tick", e)
        return
    tasks = state.get("tasks") or []
    now_ms = int(time.time() * 1000)
    to_stop = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        stop_at = t.get("_scheduledStopAtMs")
        if not isinstance(stop_at, (int, float)):
            continue
        if stop_at > now_ms:
            continue
        if (t.get("status") or "").lower() in {"stopped", "ended", "killed", "crashed"}:
            continue
        sid = t.get("sessionId") or t.get("id")
        if not sid:
            continue
        to_stop.append((sid, t.get("remoteHost") or "", t.get("sshPort") or ""))
    if not to_stop:
        return
    # Re-read state once before writing so we capture any updates from
    # concurrent UI syncs.
    stopped_any = False
    successfully_stopped_sids = set()
    for sid, host, port in to_stop:
        ok = await _stop_serve(sid, host, port)
        logger.info(f"cookbook_serve_lifecycle: stop {sid} (host={host or 'local'}): {'ok' if ok else 'failed'}")
        if ok:
            stopped_any = True
            successfully_stopped_sids.add(sid)
            # Drop the auto-registered endpoint so the model picker and
            # the chat router don't keep pointing at a dead server.
            for t in tasks:
                if isinstance(t, dict) and (t.get("sessionId") == sid or t.get("id") == sid):
                    if t.get("type") == "serve":
                        await _delete_endpoint_for_task(t)
                    t["status"] = "stopped"
                    t["_scheduledStopAtMs"] = None
                    t["_lastStatusFlipAt"] = now_ms
                    break
    if stopped_any:
        try:
            from core.atomic_io import atomic_write_json
            # Re-read the state file so concurrent UI writes (task adds,
            # status flips, config edits) are not silently overwritten.
            # Apply only our stop mutations to the fresh snapshot.
            try:
                fresh = json.loads(state_path.read_text(encoding="utf-8"))
                fresh_tasks = fresh.get("tasks") or []
            except Exception:
                fresh = state
                fresh_tasks = tasks
            for ft in fresh_tasks:
                if not isinstance(ft, dict):
                    continue
                ft_sid = ft.get("sessionId") or ft.get("id")
                if ft_sid in successfully_stopped_sids:
                    ft["status"] = "stopped"
                    ft["_scheduledStopAtMs"] = None
                    ft["_lastStatusFlipAt"] = now_ms
            fresh["tasks"] = fresh_tasks
            atomic_write_json(state_path, fresh)
        except Exception as e:
            logger.warning(f"cookbook_serve_lifecycle: state write failed: {e}")


async def cookbook_serve_lifecycle_loop() -> None:
    """Forever-loop. Registered as a startup task in app.py."""
    # Reclaim last generation's orphaned serves immediately (before the settle
    # sleep) — a server restart leaves detached serves running while the registry
    # reconciles to empty. Offloaded to a thread: the sweep shells out to `ps`
    # and may block briefly on SIGKILL escalation.
    try:
        await asyncio.to_thread(reconcile_orphans_on_boot)
    except Exception as e:
        logger.warning(f"cookbook_serve_lifecycle: boot reconcile failed: {e}")
    await asyncio.sleep(20)  # let the rest of startup settle
    while True:
        try:
            await _tick()
        except Exception as e:
            logger.warning(f"cookbook_serve_lifecycle tick failed: {e}")
        try:
            await _reap_mlx_serves()
        except Exception as e:
            logger.warning(f"cookbook_serve_lifecycle mlx reap failed: {e}")
        await asyncio.sleep(60)
