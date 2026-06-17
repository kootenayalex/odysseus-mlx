"""MLX gateway — one stable OpenAI/Ollama endpoint in front of the MLX serves.

Absorbs Baton's gateway role (P3). External consumers (OpenCode, the tailnet
sidecar) get a single base URL instead of N per-model ports:

    OpenAI:  <host>:7860/mlx/v1/{models,chat/completions,completions,embeddings}
    Ollama:  <host>:7860/mlx/api/{tags,ps,version,chat,generate,embeddings}

A request's `model` is resolved against the scheduler registry (the serves the
cookbook launched) by exact repo id, short name, or a configurable alias map,
then proxied to that serve's local mlx_lm.server port. Responses pass through
the tool-call lifter so Qwen-family models work with tool-calling clients.
Each routed request bumps the serve's last_used so the P2 idle-TTL reaper only
unloads genuinely idle models.

AUTH: this router's paths are exempt from the app's session-cookie auth (see
app.py AUTH_EXEMPT_PREFIXES), so it enforces its own: loopback is trusted
(host-local consumers); non-loopback (tailnet) requires a Bearer token equal
to ODYSSEUS_MLX_GATEWAY_KEY when that env var is set, and is refused otherwise.

PUBLIC FORK: no personal/infra literals. Host/port come from the registry;
the optional gateway key and alias map come from env.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from collections import defaultdict

import httpx
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from services import mlx_scheduler as ms
from services.mlx_tool_calls import (
    classify_stream,
    extract_tool_calls,
    normalize_tool_calls,
    ollama_messages_to_openai,
    tool_names,
)

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
# Proxy/forwarding headers that mean "this didn't really originate on loopback".
_PROXY_FWD_HEADERS = ("x-forwarded-for", "x-forwarded-host", "x-real-ip",
                      "cf-connecting-ip", "forwarded")


def _alias_map() -> dict:
    """Optional name->repo_id aliases, JSON in ODYSSEUS_MLX_ALIASES. Lets a
    consumer keep a short `model` string (e.g. {"coder": "mlx-community/...4bit"})."""
    raw = os.environ.get("ODYSSEUS_MLX_ALIASES")
    if not raw:
        return {}
    try:
        d = json.loads(raw)
        return d if isinstance(d, dict) else {}
    except ValueError:
        return {}


def _is_loopback(request: Request) -> bool:
    host = request.client.host if request.client else None
    if host not in _LOOPBACK_HOSTS:
        return False
    return not any(request.headers.get(h) for h in _PROXY_FWD_HEADERS)


def _check_auth(request: Request, authorization: str) -> None:
    if _is_loopback(request):
        return
    key = os.environ.get("ODYSSEUS_MLX_GATEWAY_KEY", "").strip()
    token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if not key or token != key:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _running_serves() -> list:
    reg = ms.load_registry()
    return [s for s in reg.values() if s.status in ("running", "loading") and s.port]


def _resolve(model: str):
    """Resolve a request `model` to a running serve. Returns the LoadedServe.

    Match order: configured alias -> exact repo id -> short name (last path
    segment) -> 404. Raises 404 with the list of currently-loaded models so a
    misrouted client gets an actionable error.
    """
    serves = _running_serves()
    if not serves:
        raise HTTPException(status_code=503, detail="no MLX models are currently loaded")
    # Resolve aliases: explicit ODYSSEUS_MLX_ALIASES first, then autoserve names
    # (so a picked "coder" maps to its repo id and hits the already-loaded serve).
    target = _alias_map().get(model)
    if target is None:
        _auto = _autoserve_config().get(model)
        target = _auto["repo_id"] if _auto else model
    by_repo = {s.repo_id: s for s in serves}
    if target in by_repo:
        return by_repo[target]
    short = {s.repo_id.split("/")[-1]: s for s in serves}
    if target in short:
        return short[target]
    if model in short:
        return short[model]
    loaded = sorted({s.repo_id for s in serves})
    raise HTTPException(status_code=404,
                        detail={"error": f"unknown model '{model}'", "loaded": loaded})


def _base_url(serve) -> str:
    host = serve.remote_host.split("@")[-1] if serve.remote_host else "127.0.0.1"
    return f"http://{host}:{serve.port}/v1"


# ── On-demand auto-serve (absorbs Baton's ensure_loaded) ──────────────────
# Per-name locks so two concurrent requests for the same model launch it once.
_autoserve_locks: dict = defaultdict(asyncio.Lock)


def _autoserve_config() -> dict:
    """Map of auto-servable name -> normalized spec dict {repo_id, venv_bin, ctx,
    priority, pin, trust_remote}. Read from MLX_AUTOSERVE_FILE (data dir, not
    committed). A bare string value is treated as the repo_id."""
    from src.constants import MLX_AUTOSERVE_FILE

    try:
        with open(MLX_AUTOSERVE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}
    out = {}
    for name, spec in (raw or {}).items():
        if isinstance(spec, str):
            spec = {"repo_id": spec}
        if isinstance(spec, dict) and spec.get("repo_id"):
            out[name] = spec
    return out


def _pick_free_port(base: int = 8130, span: int = 60) -> int:
    for port in range(base, base + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise HTTPException(status_code=503, detail="no free port for auto-serve")


def _build_mlx_cmd(spec: dict, port: int) -> str:
    venv_bin = spec.get("venv_bin") or os.environ.get("ODYSSEUS_MLX_VENV_BIN", "")
    binpath = (venv_bin.rstrip("/") + "/mlx_lm.server") if venv_bin else "mlx_lm.server"
    cmd = f"{binpath} --model {spec['repo_id']} --host 127.0.0.1 --port {port}"
    if spec.get("trust_remote"):
        cmd += " --trust-remote-code"
    return cmd


async def _probe_ready(port: int, timeout_s: int = 240) -> bool:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=5) as client:
        while time.time() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.5)
    return False


async def _ensure_served(name: str):
    """Launch an auto-servable model on a resolve-miss, wait until it's ready,
    and return its LoadedServe. Returns None if the name isn't auto-servable.

    Reuses the cookbook serve route (internal call) so admission/eviction,
    tmux launch, endpoint registration, and scheduler bookkeeping all happen
    exactly as a manual serve would."""
    from core.constants import internal_api_base
    from core.middleware import INTERNAL_TOOL_HEADER, INTERNAL_TOOL_TOKEN

    cfg = _autoserve_config()
    target = _alias_map().get(name, name)
    spec = cfg.get(name) or cfg.get(target)
    if not spec:
        return None

    async with _autoserve_locks[spec["repo_id"]]:
        # Re-check under the lock: a concurrent request may have served it.
        for s in _running_serves():
            if s.repo_id == spec["repo_id"]:
                return s
        port = _pick_free_port()
        body = {"repo_id": spec["repo_id"], "cmd": _build_mlx_cmd(spec, port)}
        if spec.get("priority") is not None:
            body["priority"] = int(spec["priority"])
        if spec.get("pin"):
            body["pin"] = True
        if spec.get("ttl_minutes") is not None:
            body["ttl_minutes"] = int(spec["ttl_minutes"])
        headers = {INTERNAL_TOOL_HEADER: INTERNAL_TOOL_TOKEN}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(f"{internal_api_base()}/api/model/serve",
                                  json=body, headers=headers)
        if r.status_code >= 400 or not (r.json() or {}).get("ok"):
            detail = (r.json() or {}).get("error") if r.content else f"HTTP {r.status_code}"
            raise HTTPException(status_code=503, detail=f"auto-serve failed: {detail}")
        if not await _probe_ready(port):
            raise HTTPException(status_code=504, detail=f"auto-served model '{name}' did not become ready")
        for s in _running_serves():
            if s.repo_id == spec["repo_id"] and s.port == port:
                return s
        # Registry race: the serve is up but its record didn't land — resolve by repo.
        for s in _running_serves():
            if s.repo_id == spec["repo_id"]:
                return s
        return None


def _advertised_models() -> list:
    """Model ids the gateway exposes to clients: every auto-servable name (so the
    model picker shows them even when nothing is loaded — Baton parity) plus any
    currently-loaded serve not covered by an autoserve alias. Auto-serve aliases
    win, so a freshly-launched model still shows under its friendly name."""
    names = list(_autoserve_config().keys())
    covered = {spec["repo_id"] for spec in _autoserve_config().values()}
    for s in _running_serves():
        if s.repo_id not in covered and s.repo_id not in names:
            names.append(s.repo_id)
    return names


def setup_mlx_gateway_routes() -> APIRouter:
    router = APIRouter(prefix="/mlx", tags=["mlx-gateway"])

    async def _resolve_and_touch(model: str):
        try:
            serve = _resolve(model)
        except HTTPException as e:
            # Not loaded — try on-demand auto-serve (Baton parity). Use the serve
            # _ensure_served returns directly: re-resolving by an autoserve alias
            # (e.g. "coder") would 404 since the loaded serve is keyed by repo id.
            if e.status_code in (404, 503):
                serve = await _ensure_served(model)
                if serve is None:
                    raise
            else:
                raise
        try:
            ms.touch_serve(serve.session_id)
        except Exception:
            pass
        return serve

    # ── OpenAI surface ────────────────────────────────────────────────────
    @router.get("/v1/models")
    async def list_models(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        return {"object": "list", "data": [
            {"id": mid, "object": "model", "owned_by": "odysseus-mlx"}
            for mid in _advertised_models()]}

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        model = body.get("model", "")
        serve = await _resolve_and_touch(model)
        # mlx_lm.server resolves the request `model` against its own --model id;
        # a short name/alias makes it try to load a new (nonexistent) HF repo.
        # Rewrite to the serve's real repo id before forwarding.
        body["model"] = serve.repo_id
        url = f"{_base_url(serve)}/chat/completions"
        names = tool_names(body)

        if body.get("stream") and not body.get("tools"):
            # No tools -> nothing to lift; raw passthrough keeps streaming cheap
            # and preserves first-token latency.
            async def gen_raw():
                client = httpx.AsyncClient(timeout=None)
                try:
                    async with client.stream("POST", url, json=body) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
                finally:
                    await client.aclose()
            return StreamingResponse(gen_raw(), media_type="text/event-stream")

        if body.get("stream"):
            # Tools offered: MLX backends emit calls as plain text, so stream
            # prose through but buffer anything that looks like a call and
            # re-emit it as a proper OpenAI tool_calls delta once complete.
            async def gen():
                client = httpx.AsyncClient(timeout=None)
                cid = "chatcmpl_" + uuid.uuid4().hex
                created = int(time.time())
                acc, fwd, mode = "", 0, "scan"
                sent_role = fwd_tool = False

                def chunk(delta, finish=None):
                    return ("data: " + json.dumps({
                        "id": cid, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }) + "\n\n").encode()

                def content_delta(text):
                    nonlocal sent_role
                    d = {"content": text}
                    if not sent_role:
                        d["role"] = "assistant"
                        sent_role = True
                    return chunk(d)

                try:
                    async with client.stream("POST", url, json=body) as r:
                        async for line in r.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            obj = None
                            try:
                                obj = json.loads(payload)
                            except ValueError:
                                continue
                            d0 = (obj.get("choices") or [{}])[0].get("delta") or {}
                            if d0.get("tool_calls"):       # backend already structured it
                                yield chunk({**({} if sent_role else {"role": "assistant"}),
                                             "tool_calls": d0["tool_calls"]})
                                sent_role = fwd_tool = True
                                continue
                            piece = d0.get("content")
                            if piece:
                                acc += piece
                            if mode == "scan":
                                mode, safe = classify_stream(acc, names)
                                if safe > fwd:
                                    yield content_delta(acc[fwd:safe])
                                    fwd = safe
                            elif mode == "prose" and len(acc) > fwd:
                                yield content_delta(acc[fwd:])
                                fwd = len(acc)
                            # mode == "tool": buffer silently until the stream ends

                    if mode == "tool":
                        tcs, _ = extract_tool_calls(acc, names)
                        if tcs:
                            for i, tc in enumerate(tcs):
                                yield chunk({"tool_calls": [{"index": i, "id": tc["id"],
                                                             "type": "function", "function": tc["function"]}]})
                            yield chunk({}, finish="tool_calls")
                        else:                              # false alarm: flush as content
                            if len(acc) > fwd:
                                yield content_delta(acc[fwd:])
                            yield chunk({}, finish="stop")
                    else:
                        if len(acc) > fwd:
                            yield content_delta(acc[fwd:])
                        yield chunk({}, finish="tool_calls" if fwd_tool else "stop")
                    yield b"data: [DONE]\n\n"
                finally:
                    await client.aclose()

            return StreamingResponse(gen(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, json=body)
        data = normalize_tool_calls(r.json(), names) if r.status_code == 200 else r.json()
        return JSONResponse(data, status_code=r.status_code)

    @router.post("/v1/completions")
    async def completions(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        serve = await _resolve_and_touch(body.get("model", ""))
        body["model"] = serve.repo_id
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_base_url(serve)}/completions", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    @router.post("/v1/embeddings")
    async def embeddings(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        serve = await _resolve_and_touch(body.get("model", ""))
        body["model"] = serve.repo_id
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_base_url(serve)}/embeddings", json=body)
        return JSONResponse(r.json(), status_code=r.status_code)

    # ── Ollama-native surface (consumers migrate by base-URL swap) ─────────
    @router.get("/api/version")
    async def api_version(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        return {"version": "odysseus-mlx-gateway"}

    @router.get("/api/tags")
    async def api_tags(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        return {"models": [
            {"name": mid, "model": mid, "size": 0, "digest": "",
             "modified_at": "2026-01-01T00:00:00Z", "details": {"family": "mlx"}}
            for mid in _advertised_models()]}

    @router.get("/api/ps")
    async def api_ps(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        return {"models": [
            {"name": s.repo_id, "model": s.repo_id, "size": (s.footprint_mb or 0) * 1_000_000}
            for s in _running_serves()]}

    @router.post("/api/chat")
    async def api_chat(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        model = body.get("model", "")
        serve = await _resolve_and_touch(model)
        oai = {"model": serve.repo_id,
               "messages": ollama_messages_to_openai(body.get("messages", [])),
               "max_tokens": (body.get("options") or {}).get("num_predict", 1024),
               "stream": False}
        if body.get("tools"):
            oai["tools"] = body["tools"]
        if body.get("format") == "json":
            oai["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_base_url(serve)}/chat/completions", json=oai)
        out = {"role": "assistant", "content": ""}
        if r.status_code == 200:
            m = (normalize_tool_calls(r.json(), tool_names(body)).get("choices") or [{}])[0].get("message", {})
            out["content"] = m.get("content") or ""
            if m.get("tool_calls"):
                out["tool_calls"] = m["tool_calls"]
        return JSONResponse({"model": model, "created_at": "2026-01-01T00:00:00Z",
                             "message": out, "done": True}, status_code=r.status_code)

    @router.post("/api/generate")
    async def api_generate(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        model = body.get("model", "")
        serve = await _resolve_and_touch(model)
        msg = {"role": "user", "content": body.get("prompt", "")}
        if body.get("images"):
            msg["images"] = body["images"]
        oai = {"model": serve.repo_id, "messages": ollama_messages_to_openai([msg]),
               "max_tokens": (body.get("options") or {}).get("num_predict", 1024), "stream": False}
        if body.get("format") == "json":
            oai["response_format"] = {"type": "json_object"}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_base_url(serve)}/chat/completions", json=oai)
        content = ""
        if r.status_code == 200:
            content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "")
        return JSONResponse({"model": model, "created_at": "2026-01-01T00:00:00Z",
                             "response": content, "done": True}, status_code=r.status_code)

    @router.post("/api/embeddings")
    async def api_embeddings(request: Request, authorization: str = Header(default="")):
        _check_auth(request, authorization)
        body = await request.json()
        model = body.get("model", "")
        serve = await _resolve_and_touch(model)
        inp = body.get("input") or body.get("prompt") or ""
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(f"{_base_url(serve)}/embeddings",
                                  json={"model": serve.repo_id, "input": inp})
        if r.status_code == 200:
            embs = [d["embedding"] for d in r.json().get("data", [])]
            return JSONResponse({"embedding": embs[0] if embs else [], "embeddings": embs})
        return JSONResponse(r.json(), status_code=r.status_code)

    return router
