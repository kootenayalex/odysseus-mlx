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

import json
import os
import time
import uuid

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
    target = _alias_map().get(model, model)
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


def setup_mlx_gateway_routes() -> APIRouter:
    router = APIRouter(prefix="/mlx", tags=["mlx-gateway"])

    async def _resolve_and_touch(model: str):
        serve = _resolve(model)
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
            {"id": s.repo_id, "object": "model", "owned_by": "odysseus-mlx"}
            for s in _running_serves()]}

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
            {"name": s.repo_id, "model": s.repo_id,
             "size": (s.footprint_mb or 0) * 1_000_000, "digest": "",
             "modified_at": "2026-01-01T00:00:00Z", "details": {"family": "mlx"}}
            for s in _running_serves()]}

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
