"""MLX gateway resolution + auth (the routing logic that fronts the serves).

The gateway maps a request `model` to a running scheduler-tracked serve and
proxies to its local port. These cover resolution (alias / repo / short name /
not-found) and the loopback/Bearer auth, without a live HTTP server.
"""

import socket

import pytest
from fastapi import HTTPException

import routes.mlx_gateway_routes as gw
from services.mlx_scheduler import LoadedServe


def _reg(monkeypatch, *serves):
    reg = {s.session_id: s for s in serves}
    monkeypatch.setattr(gw.ms, "load_registry", lambda: reg)


def _serve(sid, repo, port, status="running"):
    return LoadedServe(session_id=sid, repo_id=repo, port=port, status=status)


# --- resolution ----------------------------------------------------------- #
def test_resolve_exact_repo(monkeypatch):
    _reg(monkeypatch, _serve("s1", "mlx-community/Qwen2.5-Coder-7B-4bit", 8001))
    s = gw._resolve("mlx-community/Qwen2.5-Coder-7B-4bit")
    assert s.port == 8001


def test_resolve_short_name(monkeypatch):
    _reg(monkeypatch, _serve("s1", "mlx-community/Qwen2.5-Coder-7B-4bit", 8001))
    s = gw._resolve("Qwen2.5-Coder-7B-4bit")
    assert s.port == 8001


def test_resolve_via_alias(monkeypatch):
    _reg(monkeypatch, _serve("s1", "mlx-community/Qwen2.5-Coder-7B-4bit", 8001))
    monkeypatch.setenv("ODYSSEUS_MLX_ALIASES", '{"coder": "mlx-community/Qwen2.5-Coder-7B-4bit"}')
    s = gw._resolve("coder")
    assert s.port == 8001


def test_resolve_unknown_404(monkeypatch):
    _reg(monkeypatch, _serve("s1", "mlx-community/A-7B", 8001))
    with pytest.raises(HTTPException) as ei:
        gw._resolve("nonexistent")
    assert ei.value.status_code == 404
    assert "mlx-community/A-7B" in ei.value.detail["loaded"]


def test_resolve_none_loaded_503(monkeypatch):
    _reg(monkeypatch)
    with pytest.raises(HTTPException) as ei:
        gw._resolve("anything")
    assert ei.value.status_code == 503


def test_resolve_skips_non_running(monkeypatch):
    _reg(monkeypatch, _serve("s1", "mlx-community/A-7B", 8001, status="stopped"))
    with pytest.raises(HTTPException) as ei:
        gw._resolve("mlx-community/A-7B")
    assert ei.value.status_code == 503  # nothing actually running


def test_base_url_local():
    assert gw._base_url(_serve("s1", "x/y", 8123)) == "http://127.0.0.1:8123/v1"


def test_base_url_remote():
    s = LoadedServe(session_id="s1", repo_id="x/y", port=8123, remote_host="user@box")
    assert gw._base_url(s) == "http://box:8123/v1"


# --- alias map parsing ---------------------------------------------------- #
def test_alias_map_empty(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_ALIASES", raising=False)
    assert gw._alias_map() == {}


def test_alias_map_bad_json(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_ALIASES", "not json")
    assert gw._alias_map() == {}


# --- auth ----------------------------------------------------------------- #
class _Req:
    def __init__(self, host, headers=None):
        self.client = type("C", (), {"host": host})()
        self.headers = headers or {}


def test_loopback_trusted():
    gw._check_auth(_Req("127.0.0.1"), "")  # no raise


def test_loopback_with_proxy_header_not_trusted(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_GATEWAY_KEY", raising=False)
    with pytest.raises(HTTPException):
        gw._check_auth(_Req("127.0.0.1", {"x-forwarded-for": "1.2.3.4"}), "")


def test_nonloopback_requires_key(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_GATEWAY_KEY", raising=False)
    with pytest.raises(HTTPException):
        gw._check_auth(_Req("100.64.0.5"), "")


def test_nonloopback_with_correct_bearer(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_GATEWAY_KEY", "sekret")
    gw._check_auth(_Req("100.64.0.5"), "Bearer sekret")  # no raise


def test_nonloopback_with_wrong_bearer(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_GATEWAY_KEY", "sekret")
    with pytest.raises(HTTPException):
        gw._check_auth(_Req("100.64.0.5"), "Bearer nope")


# --- auto-serve helpers --------------------------------------------------- #
def test_autoserve_config_normalizes(monkeypatch, tmp_path):
    import json as _json
    import src.constants as const
    f = tmp_path / "mlx_autoserve.json"
    f.write_text(_json.dumps({
        "coder": {"repo_id": "mlx-community/Coder-7B-4bit", "pin": True},
        "chat": "mlx-community/Chat-4bit",          # bare string -> repo_id
        "bad": {"no_repo": 1},                        # dropped
    }))
    monkeypatch.setattr(const, "MLX_AUTOSERVE_FILE", str(f))
    cfg = gw._autoserve_config()
    assert cfg["coder"]["repo_id"] == "mlx-community/Coder-7B-4bit"
    assert cfg["coder"]["pin"] is True
    assert cfg["chat"]["repo_id"] == "mlx-community/Chat-4bit"
    assert "bad" not in cfg


def test_autoserve_config_missing_file(monkeypatch, tmp_path):
    import src.constants as const
    monkeypatch.setattr(const, "MLX_AUTOSERVE_FILE", str(tmp_path / "nope.json"))
    assert gw._autoserve_config() == {}


def test_build_mlx_cmd_with_venv_and_trust():
    cmd = gw._build_mlx_cmd({"repo_id": "x/y", "venv_bin": "/v/bin", "trust_remote": True}, 8131)
    assert cmd == "/v/bin/mlx_lm.server --model x/y --host 127.0.0.1 --port 8131 --trust-remote-code"


def test_build_mlx_cmd_bare_binary(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_VENV_BIN", raising=False)
    cmd = gw._build_mlx_cmd({"repo_id": "x/y"}, 8000)
    assert cmd == "mlx_lm.server --model x/y --host 127.0.0.1 --port 8000"


def test_build_mlx_cmd_env_venv(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_VENV_BIN", "/env/bin")
    cmd = gw._build_mlx_cmd({"repo_id": "x/y"}, 8000)
    assert cmd.startswith("/env/bin/mlx_lm.server ")


def test_pick_free_port_returns_bindable():
    port = gw._pick_free_port()
    # Should be bindable right now (nothing holding it).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
