"""Live HF-search route + rank_models(models=) injection.

Confirms an on-demand HF result is ranked through the same fit logic as the
catalog (so MLX surfaces on Metal and rows carry the catalog shape), without
hitting the network (HfApi is mocked).
"""

import sys
import types

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.hwfit.fit import rank_models
from routes.hwfit_routes import setup_hwfit_routes
import routes.hwfit_routes as hr


def _metal_system():
    return {
        "has_gpu": True, "backend": "metal", "gpu_name": "Apple M4 Pro",
        "gpu_vram_gb": 18.0, "gpu_count": 1, "available_ram_gb": 16.0,
        "total_ram_gb": 24.0, "unified_memory": True,
    }


def test_rank_models_accepts_injected_list():
    entries = [{
        "name": "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        "provider": "mlx-community", "parameter_count": "7.0B",
        "parameters_raw": 7_000_000_000, "quantization": "mlx-4bit",
        "context_length": 32768, "min_vram_gb": 4.4,
    }]
    out = rank_models(_metal_system(), models=entries, limit=10)
    assert any(r["name"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" for r in out)


class _MI:
    def __init__(self, mid):
        self.id = mid
        self.tags = []
        self.downloads = 10
        self.likes = 1
        self.pipeline_tag = "text-generation"
        self.created_at = None


def _client(monkeypatch, infos):
    # Mock detect_system -> Metal, and huggingface_hub.HfApi.list_models -> infos.
    monkeypatch.setattr(hr, "_HF_SEARCH_CACHE", {})
    import services.hwfit.hardware as hw
    monkeypatch.setattr(hw, "detect_system", lambda **k: _metal_system())

    fake_hub = types.ModuleType("huggingface_hub")

    class _Api:
        def list_models(self, **kwargs):
            return infos
    fake_hub.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    app = FastAPI()
    app.include_router(setup_hwfit_routes())
    return TestClient(app)


def test_hf_search_returns_catalog_shape_and_surfaces_mlx(monkeypatch):
    infos = [_MI("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"),
             _MI("mlx-community/Qwen3-8B-4bit")]
    client = _client(monkeypatch, infos)
    r = client.get("/api/hwfit/hf-search", params={"q": "qwen", "mlx_only": "true"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("source") == "huggingface"
    names = [m["name"] for m in data["models"]]
    assert "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit" in names
    # rows carry the catalog/fit shape
    row = data["models"][0]
    assert "quant" in row or "quantization" in row
    assert "fit_level" in row or "score" in row


def test_hf_search_mlx_only_filters_non_mlx(monkeypatch):
    infos = [_MI("mlx-community/Qwen3-8B-4bit"), _MI("meta-llama/Llama-3.1-8B-Instruct")]
    client = _client(monkeypatch, infos)
    r = client.get("/api/hwfit/hf-search", params={"q": "llama", "mlx_only": "true"})
    names = [m["name"] for m in r.json()["models"]]
    assert "meta-llama/Llama-3.1-8B-Instruct" not in names


def test_hf_search_empty_query(monkeypatch):
    client = _client(monkeypatch, [])
    r = client.get("/api/hwfit/hf-search", params={"q": ""})
    assert r.status_code == 200
    assert r.json()["models"] == []
