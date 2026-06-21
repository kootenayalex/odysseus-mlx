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


def test_build_mlx_cmd_rapid_default(monkeypatch):
    # Default engine is Rapid-MLX: positional model, served-name = repo_id,
    # native tool-calling (auto parser), prefix cache, thinking off.
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    monkeypatch.setenv("ODYSSEUS_RAPID_MLX_BIN", "/r/rapid-mlx")
    cmd = gw._build_mlx_cmd({"repo_id": "x/y"}, 8000)
    assert cmd.startswith("/r/rapid-mlx serve x/y --served-model-name x/y --host 127.0.0.1 --port 8000")
    assert "--enable-prefix-cache" in cmd
    assert "--enable-auto-tool-choice --tool-call-parser auto" in cmd
    assert "--no-thinking" in cmd


def test_build_mlx_cmd_rapid_parser_and_thinking(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    monkeypatch.setenv("ODYSSEUS_RAPID_MLX_BIN", "/r/rapid-mlx")
    cmd = gw._build_mlx_cmd({"repo_id": "x/y", "tool_call_parser": "qwen3", "thinking": True}, 8000)
    assert "--tool-call-parser qwen3" in cmd
    assert "--no-thinking" not in cmd


def test_build_mlx_cmd_mlxlm_rollback(monkeypatch):
    # ODYSSEUS_MLX_ENGINE=mlxlm restores the legacy mlx_lm.server command.
    monkeypatch.setenv("ODYSSEUS_MLX_ENGINE", "mlxlm")
    cmd = gw._build_mlx_cmd({"repo_id": "x/y", "venv_bin": "/v/bin", "trust_remote": True}, 8131)
    assert cmd == "/v/bin/mlx_lm.server --model x/y --host 127.0.0.1 --port 8131 --trust-remote-code"


def test_build_mlx_cmd_mlxlm_bare_binary(monkeypatch):
    monkeypatch.setenv("ODYSSEUS_MLX_ENGINE", "mlxlm")
    monkeypatch.delenv("ODYSSEUS_MLX_VENV_BIN", raising=False)
    cmd = gw._build_mlx_cmd({"repo_id": "x/y"}, 8000)
    assert cmd == "mlx_lm.server --model x/y --host 127.0.0.1 --port 8000"


def test_build_mlx_cmd_no_cloud_by_default(monkeypatch):
    # Sensitivity gate: a serve without cloud_model never escalates.
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    monkeypatch.setenv("ODYSSEUS_RAPID_MLX_BIN", "/r/rapid-mlx")
    cmd = gw._build_mlx_cmd({"repo_id": "x/y"}, 8000)
    assert "--cloud-model" not in cmd


def test_build_mlx_cmd_cloud_escalation(monkeypatch):
    # Provider-agnostic cloud routing: litellm model + base + threshold; the key
    # comes from the env var named by cloud_api_key_env (kept out of the spec).
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    monkeypatch.setenv("ODYSSEUS_RAPID_MLX_BIN", "/r/rapid-mlx")
    monkeypatch.setenv("MY_CLOUD_KEY", "sk-secret")
    spec = {
        "repo_id": "x/y",
        "cloud_model": "anthropic/claude-sonnet-4-5",
        "cloud_api_base": "https://api.example.com/v1",
        "cloud_threshold": 20000,
        "cloud_api_key_env": "MY_CLOUD_KEY",
    }
    cmd = gw._build_mlx_cmd(spec, 8000)
    assert "--cloud-model anthropic/claude-sonnet-4-5" in cmd
    assert "--cloud-api-base https://api.example.com/v1" in cmd
    assert "--cloud-threshold 20000" in cmd
    assert "--cloud-api-key sk-secret" in cmd


def test_build_mlx_cmd_cloud_without_key_env(monkeypatch):
    # cloud_model set but the key env var is unset → no --cloud-api-key emitted
    # (rapid-mlx can still read a litellm provider env var on the serve).
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    monkeypatch.setenv("ODYSSEUS_RAPID_MLX_BIN", "/r/rapid-mlx")
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cmd = gw._build_mlx_cmd(
        {"repo_id": "x/y", "cloud_model": "openai/gpt-4o", "cloud_api_key_env": "MISSING_KEY"}, 8000
    )
    assert "--cloud-model openai/gpt-4o" in cmd
    assert "--cloud-api-key" not in cmd


# --- whisper / STT engine ------------------------------------------------- #
def test_is_whisper_name():
    assert gw._is_whisper_name("whisper")
    assert gw._is_whisper_name("whisper-1")
    assert gw._is_whisper_name("mlx-community/whisper-large-v3-turbo")
    assert not gw._is_whisper_name("coder")
    assert not gw._is_whisper_name("")


def test_default_whisper_repo(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_WHISPER_REPO", raising=False)
    assert gw._default_whisper_repo() == "mlx-community/whisper-large-v3-turbo"
    monkeypatch.setenv("ODYSSEUS_MLX_WHISPER_REPO", "org/whisper-custom")
    assert gw._default_whisper_repo() == "org/whisper-custom"


def test_build_whisper_cmd_with_venv():
    cmd = gw._build_whisper_cmd(
        {"repo_id": "mlx-community/whisper-large-v3-turbo", "venv_bin": "/v/bin"}, 8134
    )
    assert cmd == (
        "/v/bin/mlx-openai-server launch "
        "--model-path mlx-community/whisper-large-v3-turbo "
        "--model-type whisper --served-model-name whisper "
        "--host 127.0.0.1 --port 8134"
    )


def test_build_whisper_cmd_bare_binary(monkeypatch):
    monkeypatch.delenv("ODYSSEUS_MLX_VENV_BIN", raising=False)
    cmd = gw._build_whisper_cmd({"repo_id": "x/whisper"}, 8000)
    assert cmd.startswith("mlx-openai-server launch --model-path x/whisper --model-type whisper ")


def test_build_serve_cmd_dispatches_on_engine(monkeypatch):
    # whisper engine -> mlx-openai-server; anything else -> Rapid-MLX (default engine)
    monkeypatch.delenv("ODYSSEUS_MLX_ENGINE", raising=False)
    w = gw._build_serve_cmd({"repo_id": "x/whisper", "engine": "whisper"}, 8000)
    assert "mlx-openai-server" in w and "--model-type whisper" in w
    t = gw._build_serve_cmd({"repo_id": "x/y"}, 8000)
    assert "rapid-mlx serve" in t and "mlx-openai-server" not in t


def test_advertised_models_includes_autoserve_when_idle(monkeypatch, tmp_path):
    """The picker must show auto-servable models even when none are loaded
    (Baton parity) — otherwise a cold gateway reports 'no models'."""
    import json as _json
    import src.constants as const
    f = tmp_path / "mlx_autoserve.json"
    f.write_text(_json.dumps({"coder": "org/Coder-4bit", "chat": "org/Chat-4bit"}))
    monkeypatch.setattr(const, "MLX_AUTOSERVE_FILE", str(f))
    _reg(monkeypatch)  # nothing loaded
    # Isolate from the real on-disk HF cache: _advertised_models() now also lists
    # downloaded MLX chat repos, so without this the assertion picks up whatever
    # models happen to be cached on the host.
    monkeypatch.setattr(gw, "_downloaded_mlx_chat_repos", lambda: [])
    assert set(gw._advertised_models()) == {"coder", "chat"}


def test_advertised_models_adds_loaded_not_in_config(monkeypatch, tmp_path):
    import json as _json
    import src.constants as const
    f = tmp_path / "mlx_autoserve.json"
    f.write_text(_json.dumps({"coder": "org/Coder-4bit"}))
    monkeypatch.setattr(const, "MLX_AUTOSERVE_FILE", str(f))
    # A loaded serve whose repo isn't an autoserve target shows by repo id;
    # an autoserve repo that's loaded stays under its friendly alias (no dupe).
    _reg(monkeypatch, _serve("s1", "org/Coder-4bit", 8001), _serve("s2", "org/Adhoc-7B", 8002))
    adv = gw._advertised_models()
    assert "coder" in adv and "org/Adhoc-7B" in adv and "org/Coder-4bit" not in adv


def test_resolve_via_autoserve_alias_to_loaded(monkeypatch, tmp_path):
    """A picked autoserve name resolves straight to its already-loaded serve."""
    import json as _json
    import src.constants as const
    f = tmp_path / "mlx_autoserve.json"
    f.write_text(_json.dumps({"coder": "org/Coder-4bit"}))
    monkeypatch.setattr(const, "MLX_AUTOSERVE_FILE", str(f))
    monkeypatch.delenv("ODYSSEUS_MLX_ALIASES", raising=False)
    _reg(monkeypatch, _serve("s1", "org/Coder-4bit", 8001))
    assert gw._resolve("coder").port == 8001


def test_pick_free_port_returns_bindable():
    port = gw._pick_free_port()
    # Should be bindable right now (nothing holding it).
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
