"""Shared HuggingFace-metadata → hwfit-catalog-entry normalization.

One source of truth for turning an HF `ModelInfo` (or repo name) into the
catalog entry shape used by `services/hwfit/fit.py` and the Cookbook UI. Used by
BOTH:
  * the offline catalog generator `scripts/add_hwfit_models.py` (bulk add), and
  * the live HF-search route `GET /api/hwfit/hf-search` (on-demand discovery),

so a model found via live search is sized/quant-labelled/backend-detected
identically to a curated catalog entry. In particular `quant_from_name` returns
`mlx-4bit/6bit/8bit` for MLX repos, which is what makes them render + serve as
MLX everywhere downstream.

Network/HF calls here are best-effort and swallow errors — callers always have a
usable (possibly minimal) entry or None.
"""

from __future__ import annotations

import json
import re
from datetime import datetime

# Tags that are not architecture names.
_GENERIC_TAGS = {
    "transformers", "safetensors", "conversational", "text-generation",
    "image-text-to-text", "text-generation-inference", "endpoints_compatible",
    "autotrain_compatible", "compressed-tensors", "gguf", "mlx", "vllm", "4-bit",
    "8-bit", "awq", "gptq", "fp8", "fp4", "nvfp4", "mxfp4", "nf4",
    "quantized", "chat",
}

# Rough bytes-per-param hints by quant (fit.py recomputes the real requirement).
_BPP = {
    "AWQ-4bit": 0.58, "GPTQ-Int4": 0.58, "mlx-4bit": 0.55, "mlx-6bit": 0.85,
    "AWQ-8bit": 1.1, "GPTQ-Int8": 1.1, "mlx-8bit": 1.1, "FP8": 1.1,
    "FP4": 0.58, "NVFP4": 0.58, "MXFP4": 0.58, "NF4": 0.58,
    "INT4": 0.58, "INT8": 1.1, "W4A16": 0.58, "W8A8": 1.1, "W8A16": 1.1,
    "Q4_K_M": 0.6,
}

_api = None


def _get_api():
    global _api
    if _api is None:
        from huggingface_hub import HfApi
        _api = HfApi()
    return _api


def parse_params(name):
    """Return (parameters_raw, active_parameters_or_None) from a repo name.
    Handles dense ("27B") and MoE ("235B-A22B") naming."""
    base = name.split("/")[-1]
    active = None
    m_active = re.search(r"-[Aa](\d+\.?\d*)[Bb](?![a-zA-Z])", base)
    if m_active:
        active = int(float(m_active.group(1)) * 1e9)
        base_wo = base[:m_active.start()] + base[m_active.end():]
    else:
        base_wo = base
    total = None
    for m in re.finditer(r"(\d+\.?\d*)[Bb](?![a-zA-Z])", base_wo):
        total = int(float(m.group(1)) * 1e9)
        break
    return total, active


def params_from_config(cfg):
    """Estimate (total, active) parameter counts from a HF config.json dict.
    Returns (None, None) when the architecture fields aren't usable."""
    if not isinstance(cfg, dict):
        return None, None
    for key in ("num_parameters", "n_params", "total_params"):
        v = cfg.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return int(v), None

    def _i(key, default=None):
        v = cfg.get(key, default)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    h = _i("hidden_size")
    L = _i("num_hidden_layers")
    if not h or not L:
        return None, None
    vocab = _i("vocab_size") or 0
    ffn = _i("intermediate_size") or (4 * h)
    n_heads = _i("num_attention_heads") or 0
    n_kv = _i("num_key_value_heads") or n_heads
    head_dim = _i("head_dim") or (h // n_heads if n_heads else h)
    q_proj = h * (n_heads * head_dim if n_heads else h)
    kv_proj = 2 * h * (n_kv * head_dim if n_kv else h)
    o_proj = (n_heads * head_dim if n_heads else h) * h
    per_layer_attn = q_proj + kv_proj + o_proj
    per_layer_dense_mlp = 3 * h * ffn
    n_experts = _i("num_experts") or _i("n_routed_experts") or 0
    n_shared = _i("n_shared_experts") or 0
    n_active = _i("num_experts_per_tok") or 0
    moe_ffn = _i("moe_intermediate_size") or ffn
    first_dense = _i("first_k_dense_replace") or 0
    if n_experts > 0 and n_active > 0:
        moe_layers = max(0, L - first_dense)
        dense_layers = L - moe_layers
        per_expert = 3 * h * moe_ffn
        total_mlp = (dense_layers * per_layer_dense_mlp
                     + moe_layers * (n_experts + n_shared) * per_expert)
        active_mlp = (dense_layers * per_layer_dense_mlp
                      + moe_layers * (n_active + n_shared) * per_expert)
    else:
        total_mlp = L * per_layer_dense_mlp
        active_mlp = total_mlp
    embed = vocab * h
    head = 0 if cfg.get("tie_word_embeddings", True) else vocab * h
    total = embed + head + L * per_layer_attn + total_mlp
    active = embed + head + L * per_layer_attn + active_mlp
    if total <= 0:
        return None, None
    if active == total or n_experts == 0:
        return int(total), None
    return int(total), int(active)


_CONFIG_CACHE = {}


def _fetch_config_json(repo_id):
    """Download and cache a repo's config.json. Returns a dict or None."""
    if repo_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[repo_id]
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename="config.json")
    except Exception:
        _CONFIG_CACHE[repo_id] = None
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        _CONFIG_CACHE[repo_id] = None
        return None
    _CONFIG_CACHE[repo_id] = cfg
    return cfg


def _base_model_tag(tags):
    for t in (tags or []):
        if t.startswith("base_model:"):
            return t.split(":")[-1]
    return None


def quant_from_name(name):
    """Infer the quant label from a repo name. MLX repos -> mlx-4bit/6bit/8bit
    (this is what makes them render + serve as MLX everywhere downstream)."""
    n = name.lower()
    if "nvfp4" in n:
        return "NVFP4"
    if "mxfp4" in n:
        return "MXFP4"
    if re.search(r"(^|[-_/])nf4($|[-_/])", n):
        return "NF4"
    if re.search(r"(^|[-_/])fp4($|[-_/])", n):
        return "FP4"
    if re.search(r"(^|[-_/])w4a16($|[-_/])", n):
        return "W4A16"
    if re.search(r"(^|[-_/])w8a8($|[-_/])", n):
        return "W8A8"
    if re.search(r"(^|[-_/])w8a16($|[-_/])", n):
        return "W8A16"
    is8 = "8bit" in n or "8-bit" in n or "int8" in n
    if "awq" in n:
        return "AWQ-8bit" if is8 else "AWQ-4bit"
    if "gptq" in n:
        return "GPTQ-Int8" if is8 else "GPTQ-Int4"
    if "mlx" in n:
        # Full-precision MLX builds are not a low-bit quant — size them as BF16
        # (they still serve as MLX via the name-based backend detection).
        if "bf16" in n or "fp16" in n or re.search(r"(^|[-_])f16($|[-_])", n):
            return "BF16"
        if is8:
            return "mlx-8bit"
        if "6bit" in n or "5bit" in n:  # 5bit ~ 6bit; mlx-5bit isn't a known bpp key
            return "mlx-6bit"
        return "mlx-4bit"
    if "fp8" in n:
        return "FP8"
    if "int4" in n or "4bit" in n or "4-bit" in n:
        return "INT4"
    if "int8" in n or "8bit" in n or "8-bit" in n:
        return "INT8"
    return "Q4_K_M"


def is_mlx_name(name):
    """True if a repo name denotes an MLX build."""
    return "mlx" in (name or "").lower()


def _arch_from_tags(tags):
    for t in (tags or []):
        if ":" in t or t in _GENERIC_TAGS:
            continue
        if re.fullmatch(r"[a-z0-9_]+", t) and any(c.isalpha() for c in t):
            return t
    return ""


def build_entry(mi, overrides=None):
    """Build a hwfit catalog entry dict from an HF ModelInfo. Returns None when
    the model can't be sized. `overrides` is an optional {field: value} dict."""
    name = mi.id
    provider = name.split("/")[0]
    total, active = parse_params(name)
    if total is None and overrides and overrides.get("parameter_count"):
        total, _ = parse_params("x/" + overrides["parameter_count"])
    if total is None:
        bm = _base_model_tag(getattr(mi, "tags", None))
        if bm:
            bt, ba = parse_params(bm)
            if bt:
                total = bt
                if ba and active is None:
                    active = ba
    quant = quant_from_name(name)
    if total is None:
        config_targets = [name]
        bm = _base_model_tag(getattr(mi, "tags", None))
        if bm and bm != name:
            config_targets.append(bm)
        for target in config_targets:
            cfg = _fetch_config_json(target)
            if not cfg:
                continue
            ct, ca = params_from_config(cfg)
            if ct:
                total = ct
                if ca and active is None:
                    active = ca
                break
    if total is None:
        try:
            full = _get_api().model_info(name, files_metadata=False)
            st = getattr(full, "safetensors", None)
            if st:
                params_by_dtype = getattr(st, "parameters", None) or {}
                if quant.endswith("4bit") or quant.endswith("Int4"):
                    pack_factor = 8
                elif quant.endswith("8bit") or quant.endswith("Int8") or quant in ("FP8", "NVFP4"):
                    pack_factor = 4
                else:
                    pack_factor = 1
                if params_by_dtype:
                    packed = sum(c for d, c in params_by_dtype.items() if d in ("I32", "I64"))
                    rest = sum(c for d, c in params_by_dtype.items() if d not in ("I32", "I64"))
                    total = packed * pack_factor + rest
                elif getattr(st, "total", None):
                    total = int(st.total) * pack_factor
        except Exception:
            pass
    if total is None:
        return None
    pb = total / 1e9
    created = getattr(mi, "created_at", None)
    rel = created.strftime("%Y-%m-%d") if created else datetime.utcnow().strftime("%Y-%m-%d")
    bpp = _BPP.get(quant, 0.6)
    vram = round(pb * bpp + 0.5, 1)
    entry = {
        "name": name,
        "provider": provider,
        "parameter_count": f"{round(pb, 1)}B",
        "parameters_raw": total,
        "min_ram_gb": max(1.0, round(vram * 0.6, 1)),
        "recommended_ram_gb": max(2.0, round(vram * 1.2, 1)),
        "min_vram_gb": vram,
        "quantization": quant,
        "context_length": 32768,
        "use_case": "General purpose",
        "capabilities": [],
        "pipeline_tag": getattr(mi, "pipeline_tag", None) or "text-generation",
        "architecture": _arch_from_tags(getattr(mi, "tags", None)),
        "hf_downloads": getattr(mi, "downloads", 0) or 0,
        "hf_likes": getattr(mi, "likes", 0) or 0,
        "release_date": rel,
        "_discovered": True,
    }
    if active:
        entry["is_moe"] = True
        entry["active_parameters"] = active
    entry.update(overrides or {})
    if overrides and "parameter_count" in overrides and "parameters_raw" not in overrides:
        t2, _ = parse_params("x/" + overrides["parameter_count"])
        if t2:
            entry["parameters_raw"] = t2
    return entry
