"""MLX serve governance — budget admission, priority/LRU eviction, idle TTL.

These lock in the Baton-derived scheduler logic ported in P2: the box has a
fixed unified-memory budget, so the cookbook must refuse or evict instead of
OOMing when MLX serves pile up. Pure-planner tests (no server, no tmux).
"""

import time

from services import mlx_scheduler as ms
from services.mlx_scheduler import LoadedServe


def _serve(sid, mb, *, priority=5, pinned=False, last_used_ms=0, ttl=600, status="running"):
    return LoadedServe(
        session_id=sid, footprint_mb=mb, priority=priority, pinned=pinned,
        last_used_ms=last_used_ms, ttl_seconds=ttl, status=status,
    )


# --- MLX detection -------------------------------------------------------- #
def test_is_mlx_cmd():
    assert ms.is_mlx_cmd("/venv/bin/mlx_lm.server --model X --port 8080")
    assert not ms.is_mlx_cmd("vllm serve X")
    assert not ms.is_mlx_cmd("llama-server -m x.gguf")
    assert not ms.is_mlx_cmd("")
    assert not ms.is_mlx_cmd(None)


def test_port_and_model_from_cmd():
    cmd = "mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-4bit --host 127.0.0.1 --port 8123"
    assert ms.port_from_cmd(cmd) == 8123
    assert ms.model_from_cmd(cmd) == "mlx-community/Qwen2.5-Coder-7B-4bit"
    assert ms.port_from_cmd("no port here") is None


# --- footprint estimation ------------------------------------------------- #
def test_params_b_from_name():
    assert ms.params_b_from_name("Qwen2.5-Coder-32B-Instruct-4bit") == 32.0
    assert ms.params_b_from_name("Qwen2.5-Coder-7B-Instruct-4bit") == 7.0
    # MoE-style tag: total wins over active
    assert ms.params_b_from_name("Qwen3-35B-A3B") == 35.0
    assert ms.params_b_from_name("some-500M-tiny") == 0.5
    assert ms.params_b_from_name("no-size-here") == 0.0


def test_quant_from_name():
    assert ms.quant_from_name("Model-8bit") == "mlx-8bit"
    assert ms.quant_from_name("Model-6bit") == "mlx-6bit"
    assert ms.quant_from_name("Model-4bit") == "mlx-4bit"
    assert ms.quant_from_name("Model-unlabeled") == "mlx-4bit"  # safe default


def test_estimate_footprint_monotonic_and_sane():
    big = ms.estimate_footprint_mb("mlx-community/Qwen2.5-Coder-32B-Instruct-4bit")
    small = ms.estimate_footprint_mb("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
    assert big > small
    # 7B @ 4-bit is well under and 32B @ 4-bit well over half the 17 GB budget.
    assert 3000 < small < 8000
    assert big > 14000


def test_estimate_footprint_unknown_name_is_conservative():
    # Unparseable size must NOT estimate ~0 (which would slip past admission).
    assert ms.estimate_footprint_mb("mystery-model") >= 4096


# --- budget --------------------------------------------------------------- #
def test_budget_env_override(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_BUDGET_MB", "12345")
    assert ms.budget_mb() == 12345


def test_budget_detected_takes_precedence_over_fallback(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_BUDGET_MB", raising=False)
    assert ms.budget_mb(detected_vram_gb=16.0) == 16384


def test_budget_fallback(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_BUDGET_MB", raising=False)
    # No detected value and (on non-Apple CI) no hardware budget -> fallback.
    # If hardware detection DOES return a value, it must still be a sane budget.
    b = ms.budget_mb()
    assert b > 0


# --- admission: loads now ------------------------------------------------- #
def test_loads_now_when_room():
    loaded = [_serve("a", 4000)]
    plan = ms.plan_admission(loaded, need_mb=5000, budget=17000)
    assert plan["decision"] == "loads_now"
    assert plan["evict"] == []
    assert plan["free_mb"] == 13000


def test_zero_footprint_loads_now():
    plan = ms.plan_admission([], need_mb=0, budget=17000)
    assert plan["decision"] == "loads_now"


# --- admission: eviction -------------------------------------------------- #
def test_evicts_lowest_priority_first():
    loaded = [
        _serve("keep", 6000, priority=9),
        _serve("low", 6000, priority=1),
        _serve("mid", 6000, priority=5),
    ]
    # Need 6000 more; budget 17000, used 18000 -> over. Must evict the lowest.
    plan = ms.plan_admission(loaded, need_mb=6000, budget=17000)
    assert plan["decision"] == "loads_after_evict"
    assert "low" in plan["evict"]
    assert "keep" not in plan["evict"]


def test_evicts_lru_within_same_priority():
    loaded = [
        _serve("fresh", 6000, priority=5, last_used_ms=2000),
        _serve("stale", 6000, priority=5, last_used_ms=1000),
    ]
    # used 12000, budget 17000, free 5000; need 6000 -> evict the LRU (6000)
    # giving 11000 free. Only the stale one needs to go.
    plan = ms.plan_admission(loaded, need_mb=6000, budget=17000)
    assert plan["decision"] == "loads_after_evict"
    assert plan["evict"] == ["stale"]


def test_pinned_never_evicted():
    loaded = [_serve("pinned", 15000, priority=1, pinned=True)]
    plan = ms.plan_admission(loaded, need_mb=8000, budget=17000)
    # Pinned can't be evicted and there's only 2000 free -> rejected.
    assert plan["decision"] == "rejected"
    assert plan["evict"] == []


def test_rejected_when_even_full_eviction_insufficient():
    loaded = [_serve("a", 5000, priority=1)]
    plan = ms.plan_admission(loaded, need_mb=20000, budget=17000)
    assert plan["decision"] == "rejected"
    # Reports the best it could do.
    assert plan["max_free_after_evict_mb"] == 17000


def test_eviction_stops_once_enough_freed():
    loaded = [
        _serve("a", 6000, priority=1, last_used_ms=1),
        _serve("b", 6000, priority=2, last_used_ms=2),
        _serve("c", 6000, priority=3, last_used_ms=3),
    ]
    # used 18000, budget 17000, need 6000 -> free=-1000, need to free >=7000.
    # Evicting "a" (6000) -> free 5000, still short; evict "b" -> 11000 free. Stop.
    plan = ms.plan_admission(loaded, need_mb=6000, budget=17000)
    assert plan["decision"] == "loads_after_evict"
    assert plan["evict"] == ["a", "b"]
    assert "c" not in plan["evict"]


# --- TTL ------------------------------------------------------------------ #
def test_expired_picks_idle_nonpinned():
    now = 10_000_000
    loaded = [
        _serve("idle", 4000, last_used_ms=now - 700_000, ttl=600),   # 700s idle > 600 ttl
        _serve("busy", 4000, last_used_ms=now - 100_000, ttl=600),   # 100s idle
        _serve("pinnedidle", 4000, last_used_ms=now - 999_000, ttl=600, pinned=True),
        _serve("nottl", 4000, last_used_ms=now - 999_000, ttl=0),
    ]
    assert ms.expired(loaded, now_ms=now) == ["idle"]


def test_expired_ignores_serves_never_used():
    now = 10_000_000
    loaded = [_serve("neverused", 4000, last_used_ms=0, ttl=600)]
    # last_used_ms == 0 means "no activity stamp yet" -> not eligible for idle reap.
    assert ms.expired(loaded, now_ms=now) == []


# --- snapshot ------------------------------------------------------------- #
def test_snapshot_accounting():
    loaded = [_serve("a", 4000), _serve("b", 5000)]
    snap = ms.snapshot(loaded, budget=17000, now_ms=0)
    assert snap["budget_mb"] == 17000
    assert snap["used_mb"] == 9000
    assert snap["free_mb"] == 8000
    assert len(snap["models"]) == 2


# --- registry round-trip -------------------------------------------------- #
def test_registry_roundtrip(tmp_path, monkeypatch):
    import src.constants as const
    reg_file = tmp_path / "mlx_scheduler.json"
    monkeypatch.setattr(const, "MLX_SCHEDULER_FILE", str(reg_file))

    assert ms.load_registry() == {}
    ms.register_serve(_serve("s1", 4000, priority=7))
    ms.register_serve(_serve("s2", 5000))
    reg = ms.load_registry()
    assert set(reg) == {"s1", "s2"}
    assert reg["s1"].footprint_mb == 4000
    assert reg["s1"].priority == 7

    ts = int(time.time() * 1000)
    ms.touch_serve("s1", now_ms=ts)
    assert ms.load_registry()["s1"].last_used_ms == ts

    assert ms.set_pinned("s2", True) is True
    assert ms.load_registry()["s2"].pinned is True
    assert ms.set_pinned("nope", True) is False

    ms.deregister_serve("s1")
    assert set(ms.load_registry()) == {"s2"}
