"""Shared HF→catalog normalization (used by the generator + live HF search).

The load-bearing guarantee: MLX repos get an `mlx-*bit` quant label so they
render and serve as MLX everywhere downstream (fit ranking, JS _detectBackend,
the serve command). Also covers param parsing and the entry shape.
"""

from services.hwfit import hf_ingest as hi


# --- quant inference ------------------------------------------------------ #
def test_quant_from_name_mlx():
    assert hi.quant_from_name("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit") == "mlx-4bit"
    assert hi.quant_from_name("lmstudio-community/Qwen3-1.7B-MLX-4bit") == "mlx-4bit"
    assert hi.quant_from_name("mlx-community/DeepSeek-Coder-V2-Lite-Instruct-6bit") == "mlx-6bit"
    assert hi.quant_from_name("mlx-community/Qwen3-8B-8bit") == "mlx-8bit"


def test_quant_from_name_non_mlx_unchanged():
    assert hi.quant_from_name("cyankiwi/Foo-AWQ-4bit") == "AWQ-4bit"
    assert hi.quant_from_name("org/Foo-GPTQ-Int8") == "GPTQ-Int8"
    assert hi.quant_from_name("org/Foo-fp8") == "FP8"
    assert hi.quant_from_name("org/Plain-7B") == "Q4_K_M"  # default


def test_is_mlx_name():
    assert hi.is_mlx_name("mlx-community/Anything")
    assert hi.is_mlx_name("lmstudio-community/X-MLX-4bit")
    assert not hi.is_mlx_name("unsloth/Qwen3-GGUF")


# --- param parsing -------------------------------------------------------- #
def test_parse_params_dense_and_moe():
    assert hi.parse_params("mlx-community/Qwen2.5-Coder-32B-Instruct-3bit") == (32_000_000_000, None)
    total, active = hi.parse_params("org/Qwen3-235B-A22B")
    assert total == 235_000_000_000 and active == 22_000_000_000
    # "4bit" must NOT be read as 4B params
    assert hi.parse_params("org/Tiny-4bit") == (None, None)


# --- build_entry shape (no network: name carries the size) ---------------- #
class _MI:
    def __init__(self, mid, tags=None, downloads=0, likes=0):
        self.id = mid
        self.tags = tags or []
        self.downloads = downloads
        self.likes = likes
        self.pipeline_tag = "text-generation"
        self.created_at = None


def test_build_entry_mlx_shape():
    e = hi.build_entry(_MI("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", downloads=123))
    assert e["name"] == "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"
    assert e["provider"] == "mlx-community"
    assert e["quantization"] == "mlx-4bit"
    assert e["parameters_raw"] == 7_000_000_000
    assert e["parameter_count"] == "7.0B"
    assert e["min_vram_gb"] > 0
    assert e["hf_downloads"] == 123


def test_build_entry_moe_flags():
    e = hi.build_entry(_MI("mlx-community/Qwen3-235B-A22B-4bit"))
    assert e["is_moe"] is True
    assert e["active_parameters"] == 22_000_000_000


def test_catalog_mlx_entries_not_mislabelled_as_gguf():
    """Guard: every MLX-named catalog entry carries an MLX/precision quant, never
    a GGUF label (Q*_K_*). A GGUF label mis-sizes the model and breaks MLX
    treatment downstream — this is the bug Part 4 fixed."""
    import json
    import re
    from services.hwfit.models import model_catalog_path
    gguf = re.compile(r"^(Q\d|IQ\d)", re.I)
    with open(model_catalog_path(), encoding="utf-8") as f:
        catalog = json.load(f)
    bad = [(m["name"], m.get("quantization")) for m in catalog
           if hi.is_mlx_name(m.get("name", "")) and gguf.match(str(m.get("quantization", "")))]
    assert bad == [], f"MLX entries mislabelled as GGUF: {bad[:5]}"


def test_build_entry_unsizable_returns_none(monkeypatch):
    # No size in the name, no config/safetensors reachable -> None (skipped).
    monkeypatch.setattr(hi, "_fetch_config_json", lambda r: None)
    monkeypatch.setattr(hi, "_get_api", lambda: (_ for _ in ()).throw(RuntimeError("no net")))
    assert hi.build_entry(_MI("mlx-community/MysteryModel-4bit")) is None
