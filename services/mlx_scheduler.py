"""MLX serve governance — memory-budget admission + priority/LRU eviction + idle TTL.

Ported in spirit from Baton's supervisor (an on-demand MLX gateway): the same
budget/admission/eviction/TTL model, adapted to how Odysseus actually serves
models. Odysseus launches each model as an external tmux session and tracks it
out-of-process, so this is NOT an in-process subprocess supervisor. Instead it
is:

  * a **pure planner** — `plan_admission()` / `select_victims()` / `expired()`
    decide, from a snapshot of currently-loaded MLX serves and a memory budget,
    whether a new serve fits, what to evict if it doesn't, and which idle serves
    have outlived their TTL. No I/O, fully unit-testable.

  * a **small persistent registry** — `load_registry()` / `save_registry()` keep
    a JSON map of session_id -> serve record next to the cookbook state, so the
    serve route and the lifecycle loop share one authoritative view of what MLX
    models are loaded and how big they are.

The serve route consults `plan_admission()` before launching an MLX model and
kills the victims it returns; the cookbook serve lifecycle loop calls
`expired()` to unload idle models. Without this, two cookbook-served MLX models
can exceed the unified-memory budget and OOM the box (we hit exactly this with
8-bit DeepSeek under Baton).

PUBLIC FORK: this module must contain no personal/infra literals. The budget is
read from the ODYSSEUS_MLX_BUDGET_MB env var, else derived from the detected
unified-memory (Metal) budget, else a conservative fallback.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

# Defaults mirror Baton's config (ttl 600s, priority 5, budget 17000MB).
DEFAULT_TTL_SECONDS = 600
DEFAULT_PRIORITY = 5
# Fallback when neither the env override nor hardware detection yields a budget.
# 17 GB matches the usable Metal working set on a 24 GB unified-memory Mac.
FALLBACK_BUDGET_MB = 17000
# Context length assumed when estimating a serve's footprint from its name.
DEFAULT_CTX = 8192


# --------------------------------------------------------------------------- #
# MLX detection
# --------------------------------------------------------------------------- #
def is_mlx_cmd(cmd: str | None) -> bool:
    """True when a serve command launches the MLX backend (mlx_lm.server)."""
    return bool(cmd) and "mlx_lm.server" in cmd


_PORT_RE = re.compile(r"--port\s+(\d+)")
_MODEL_RE = re.compile(r"--model\s+(\S+)")


def port_from_cmd(cmd: str | None) -> int | None:
    m = _PORT_RE.search(cmd or "")
    return int(m.group(1)) if m else None


def model_from_cmd(cmd: str | None) -> str | None:
    m = _MODEL_RE.search(cmd or "")
    return m.group(1) if m else None


# --------------------------------------------------------------------------- #
# Footprint estimation
# --------------------------------------------------------------------------- #
def quant_from_name(name: str | None) -> str:
    """Infer the MLX quant key (matching hwfit QUANT_BPP) from a repo/model name."""
    n = (name or "").lower()
    if re.search(r"(^|[-_])8\s*-?bit|mlx-8bit|8bit", n):
        return "mlx-8bit"
    if re.search(r"(^|[-_])6\s*-?bit|mlx-6bit|6bit", n):
        return "mlx-6bit"
    # 4-bit is the common MLX default; also the safe assumption for unlabeled.
    return "mlx-4bit"


def params_b_from_name(name: str | None) -> float:
    """Parse a parameter count (in billions) from a repo/model name, e.g.
    'Qwen2.5-Coder-32B-Instruct-4bit' -> 32.0. Returns 0.0 if none found."""
    n = name or ""
    # Prefer an explicit billions marker; pick the largest (handles A3B MoE tags
    # where total > active, e.g. '35B-A3B' -> 35).
    bills = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*[bB]\b", n)]
    if bills:
        return max(bills)
    millions = re.findall(r"(\d+(?:\.\d+)?)\s*[mM]\b", n)
    if millions:
        return max(float(x) for x in millions) / 1000.0
    return 0.0


def estimate_footprint_mb(repo_id: str | None, ctx: int = DEFAULT_CTX) -> int:
    """Estimate the memory footprint (MB) of serving an MLX model.

    Reuses hwfit's VRAM estimator (params x bytes-per-param + KV cache +
    overhead). Builds a minimal model record from the repo name; if hwfit isn't
    importable for any reason, falls back to a coarse params x bpp estimate so
    admission still has a number to work with.
    """
    quant = quant_from_name(repo_id)
    pb = params_b_from_name(repo_id)
    if pb <= 0:
        # Unknown size: assume a conservative 4 GB so an unparseable name can't
        # slip past the budget (a 0-param estimate is just the 0.5 GB overhead).
        return 4096
    try:
        from services.hwfit.models import estimate_memory_gb

        gb = estimate_memory_gb({"parameter_count": f"{pb}B"}, quant, ctx)
    except Exception:
        bpp = {"mlx-8bit": 1.0, "mlx-6bit": 0.75, "mlx-4bit": 0.55}.get(quant, 0.55)
        gb = pb * bpp + 0.000008 * pb * ctx + 0.5
    # Never admit on a zero estimate (unknown size) — assume a conservative 4 GB
    # so an unparseable name can't slip past the budget and OOM the box.
    mb = int(round(gb * 1024)) if gb > 0 else 4096
    return max(mb, 512)


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
def budget_mb(detected_vram_gb: float | None = None) -> int:
    """Resolve the MLX memory budget (MB).

    Priority: ODYSSEUS_MLX_BUDGET_MB env override > caller-supplied detected
    unified-memory budget > hardware auto-detection (Apple Silicon only) >
    conservative fallback.
    """
    env = os.environ.get("ODYSSEUS_MLX_BUDGET_MB")
    if env:
        try:
            v = int(float(env))
            if v > 0:
                return v
        except ValueError:
            pass
    if detected_vram_gb and detected_vram_gb > 0:
        return int(round(detected_vram_gb * 1024))
    try:
        from services.hwfit import hardware

        info = hardware._detect_apple_silicon()
        if info and info.get("gpu_vram_gb"):
            return int(round(float(info["gpu_vram_gb"]) * 1024))
    except Exception:
        pass
    return FALLBACK_BUDGET_MB


# --------------------------------------------------------------------------- #
# Serve records + pure planner
# --------------------------------------------------------------------------- #
@dataclass
class LoadedServe:
    """A currently-loaded MLX serve, as the planner sees it."""

    session_id: str
    repo_id: str = ""
    port: int | None = None
    footprint_mb: int = 0
    priority: int = DEFAULT_PRIORITY
    pinned: bool = False
    ttl_seconds: int = DEFAULT_TTL_SECONDS
    last_used_ms: int = 0
    remote_host: str = ""
    ssh_port: str = ""
    endpoint_id: str | None = None
    status: str = "running"

    @classmethod
    def from_record(cls, rec: dict) -> "LoadedServe":
        return cls(
            session_id=rec.get("session_id") or rec.get("sessionId") or "",
            repo_id=rec.get("repo_id", ""),
            port=rec.get("port"),
            footprint_mb=int(rec.get("footprint_mb") or 0),
            priority=int(rec.get("priority", DEFAULT_PRIORITY)),
            pinned=bool(rec.get("pinned", False)),
            ttl_seconds=int(rec.get("ttl_seconds", DEFAULT_TTL_SECONDS)),
            last_used_ms=int(rec.get("last_used_ms") or 0),
            remote_host=rec.get("remote_host", "") or "",
            ssh_port=str(rec.get("ssh_port", "") or ""),
            endpoint_id=rec.get("endpoint_id"),
            status=rec.get("status", "running"),
        )

    def to_record(self) -> dict:
        return {
            "session_id": self.session_id,
            "repo_id": self.repo_id,
            "port": self.port,
            "footprint_mb": self.footprint_mb,
            "priority": self.priority,
            "pinned": self.pinned,
            "ttl_seconds": self.ttl_seconds,
            "last_used_ms": self.last_used_ms,
            "remote_host": self.remote_host,
            "ssh_port": self.ssh_port,
            "endpoint_id": self.endpoint_id,
            "status": self.status,
        }


def used_mb(loaded: list[LoadedServe]) -> int:
    return sum(s.footprint_mb for s in loaded if s.status in ("loading", "running"))


def select_victims(
    loaded: list[LoadedServe], need_mb: int, budget: int, *, exclude: str | None = None
) -> list[str]:
    """Pick session_ids to evict (lowest priority, then least-recently-used)
    until `need_mb` would fit within `budget`, or return as many as we can.

    Mirrors Baton's _pick_victim ordering: sort by (priority asc, last_used asc)
    so low-priority + stale serves die first. Pinned serves are never evicted.
    """
    free = budget - used_mb(loaded)
    if free >= need_mb:
        return []
    cands = sorted(
        (s for s in loaded if s.session_id != exclude and not s.pinned and s.status in ("loading", "running")),
        key=lambda s: (s.priority, s.last_used_ms),
    )
    victims: list[str] = []
    freed = 0
    for s in cands:
        if free + freed >= need_mb:
            break
        victims.append(s.session_id)
        freed += s.footprint_mb
    return victims


def plan_admission(
    loaded: list[LoadedServe], need_mb: int, budget: int, *, exclude: str | None = None
) -> dict:
    """Decide whether a new serve of `need_mb` fits, mirroring Baton.admission().

    Returns a dict with `decision` in {loads_now, loads_after_evict, rejected}
    plus the eviction plan and the accounting that produced it.
    """
    free = budget - used_mb(loaded)
    base = {"need_mb": need_mb, "budget_mb": budget, "free_mb": free, "evict": []}
    if need_mb <= 0:
        return {**base, "decision": "loads_now"}
    if free >= need_mb:
        return {**base, "decision": "loads_now"}
    victims = select_victims(loaded, need_mb, budget, exclude=exclude)
    freed = sum(s.footprint_mb for s in loaded if s.session_id in set(victims))
    if free + freed >= need_mb:
        return {**base, "decision": "loads_after_evict", "evict": victims,
                "free_after_evict_mb": free + freed}
    return {**base, "decision": "rejected", "evict": victims,
            "max_free_after_evict_mb": free + freed}


def expired(loaded: list[LoadedServe], now_ms: int | None = None) -> list[str]:
    """Session_ids of non-pinned serves idle longer than their TTL."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    out = []
    for s in loaded:
        if s.pinned or s.ttl_seconds <= 0 or s.status not in ("running", "loading"):
            continue
        if s.last_used_ms and (now_ms - s.last_used_ms) > s.ttl_seconds * 1000:
            out.append(s.session_id)
    return out


def snapshot(loaded: list[LoadedServe], budget: int, now_ms: int | None = None) -> dict:
    """Baton-style snapshot for the GUI: budget/used/free + per-serve state."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    u = used_mb(loaded)
    return {
        "budget_mb": budget,
        "used_mb": u,
        "free_mb": budget - u,
        "models": [
            {
                **s.to_record(),
                "idle_seconds": round((now_ms - s.last_used_ms) / 1000, 1) if s.last_used_ms else None,
            }
            for s in loaded
        ],
    }


# --------------------------------------------------------------------------- #
# Persistent registry (thin I/O; the planner above stays pure)
# --------------------------------------------------------------------------- #
def _registry_path() -> str:
    from src.constants import MLX_SCHEDULER_FILE

    return MLX_SCHEDULER_FILE


def load_registry() -> dict[str, LoadedServe]:
    import json

    path = _registry_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (FileNotFoundError, ValueError):
        return {}
    serves = raw.get("serves") or {}
    return {sid: LoadedServe.from_record({**rec, "session_id": sid}) for sid, rec in serves.items()}


def save_registry(reg: dict[str, LoadedServe]) -> None:
    from core.atomic_io import atomic_write_json

    payload = {"serves": {sid: s.to_record() for sid, s in reg.items()}}
    atomic_write_json(_registry_path(), payload)


def register_serve(serve: LoadedServe) -> None:
    reg = load_registry()
    reg[serve.session_id] = serve
    save_registry(reg)


def deregister_serve(session_id: str) -> None:
    reg = load_registry()
    if session_id in reg:
        del reg[session_id]
        save_registry(reg)


def touch_serve(session_id: str, now_ms: int | None = None) -> None:
    """Bump last_used_ms (called by the gateway in P3 on each routed request)."""
    if now_ms is None:
        now_ms = int(time.time() * 1000)
    reg = load_registry()
    s = reg.get(session_id)
    if s:
        s.last_used_ms = now_ms
        save_registry(reg)


def set_pinned(session_id: str, pinned: bool) -> bool:
    reg = load_registry()
    s = reg.get(session_id)
    if not s:
        return False
    s.pinned = pinned
    save_registry(reg)
    return True
