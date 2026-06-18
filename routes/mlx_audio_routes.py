"""MLX audio gateway — OpenAI-compatible speech-to-text on the MLX surface.

Adds one route:

    OpenAI:  <host>:7860/mlx/v1/audio/transcriptions

Whisper is a speech model, so it can't ride mlx_lm.server like the chat/embed
serves; it runs on mlx-openai-server's whisper handler. This route reuses the
same on-demand serve machinery the chat gateway uses — it resolves a running
whisper serve, auto-serves one (admission/TTL/eviction, dynamic port) on a miss,
then reverse-proxies the multipart upload to it. So the feature is
self-contained: no standing whisper process or hard-coded port required.

Set ODYSSEUS_MLX_WHISPER_URL to bypass the scheduler and proxy to an external
whisper serve instead (e.g. one you manage yourself).

AUTH: like the rest of /mlx, this path is exempt from the app's session-cookie
auth (see app.py AUTH_EXEMPT_PREFIXES) and reuses the gateway's _check_auth —
loopback is trusted, non-loopback (tailnet) requires Bearer == ODYSSEUS_MLX_GATEWAY_KEY.

PUBLIC FORK: no personal/infra literals. The default whisper model and the
optional direct serve URL come from env.
"""

from __future__ import annotations

import os

import httpx
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from routes.mlx_gateway_routes import _base_url, _check_auth, _ensure_served, _resolve
from services import mlx_scheduler as ms
from src.upload_limits import STT_MAX_AUDIO_BYTES, read_upload_limited

# Request `model` values that mean "the default whisper serve".
_ALIASES = {"", "whisper", "whisper-1", "whisper-large-v3-turbo",
            "mlx-community/whisper-large-v3-turbo"}
# Optional escape hatch: proxy straight to an external whisper serve, skipping
# the scheduler. Empty (default) → use the on-demand scheduler-managed serve.
_DIRECT_URL = os.environ.get("ODYSSEUS_MLX_WHISPER_URL", "").strip()
_TIMEOUT = httpx.Timeout(600.0)  # a long clip on a warm serve can take minutes


async def _whisper_target(model: str) -> str:
    """Resolve the whisper serve's /audio/transcriptions URL, auto-serving on a
    miss. Honors ODYSSEUS_MLX_WHISPER_URL as a direct-proxy override."""
    if _DIRECT_URL:
        return _DIRECT_URL
    name = "whisper" if model in _ALIASES else model
    try:
        serve = _resolve(name)
    except HTTPException as e:
        if e.status_code in (404, 503):
            serve = await _ensure_served(name)
        else:
            raise
    if serve is None:
        raise HTTPException(status_code=503,
                            detail=f"whisper model '{model}' could not be served")
    try:
        ms.touch_serve(serve.session_id)
    except Exception:
        pass
    return f"{_base_url(serve)}/audio/transcriptions"


def setup_mlx_audio_routes() -> APIRouter:
    router = APIRouter(prefix="/mlx", tags=["mlx-audio"])

    @router.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: str = Form(default="whisper"),
        response_format: str = Form(default="json"),
        language: str = Form(default=""),
        authorization: str = Header(default=""),
    ):
        _check_auth(request, authorization)

        data = await read_upload_limited(file, STT_MAX_AUDIO_BYTES, "Audio file")
        url = await _whisper_target(model)

        files = {"file": (file.filename or "audio.wav", data, file.content_type or "audio/wav")}
        # The serve advertises its model under its served-model-name ("whisper").
        form = {"model": "whisper", "response_format": response_format}
        if language:
            form["language"] = language

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(url, data=form, files=files)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=503,
                                detail=f"whisper serve unreachable: {exc}") from exc

        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)

        # The serve returns JSON ({"text": ...}) regardless of response_format,
        # so normalize: response_format=text → the bare transcript, else JSON.
        try:
            payload = r.json()
        except ValueError:
            payload = None
        if response_format == "text":
            if isinstance(payload, dict) and "text" in payload:
                return PlainTextResponse(payload["text"])
            return PlainTextResponse(r.text)
        if payload is not None:
            return JSONResponse(payload)
        return PlainTextResponse(r.text)

    return router
