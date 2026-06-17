#!/usr/bin/env python3
"""
add_hwfit_models.py — bulk-add Hugging Face models to the hwfit catalog
(services/hwfit/data/hf_models.json).

Adds:
  * every model from one or more HF authors (e.g. cyankiwi's AWQ quants,
    mlx-community's MLX builds)
  * any explicitly-listed repos

Metadata normalization (param size, quant label, arch, VRAM hint) is shared
with the live HF-search route via `services/hwfit/hf_ingest.build_entry`, so a
curated catalog entry and an on-demand search result are sized/labelled the same.

Re-runnable: merges by `name`, leaving existing entries untouched unless
--overwrite is passed. Writes a .bak first.

Usage:
    python3 scripts/add_hwfit_models.py [--overwrite]
"""
import json
import os
import sys

# Allow `python3 scripts/add_hwfit_models.py` (script dir on sys.path, not repo
# root) to import the shared services.* normalization module.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from huggingface_hub import HfApi

from services.hwfit.hf_ingest import build_entry as _entry_from_modelinfo

DATA_PATH = os.path.join(_REPO_ROOT, "services", "hwfit", "data", "hf_models.json")

AUTHORS = ["cyankiwi"]

# Curated mlx-community / MLX repos to add (the long tail is reachable on demand
# via the live HF-search route). Includes the local coder set + popular families.
MLX_REPOS = [
    "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
    "mlx-community/Qwen2.5-Coder-32B-Instruct-3bit",
    "mlx-community/DeepSeek-Coder-V2-Lite-Instruct-6bit",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Qwen3-8B-4bit",
    "mlx-community/Qwen3-14B-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
]

# Specific repos to add (in addition to the authors above). Optional explicit
# overrides {repo: {field: value}} for things the name/metadata can't convey.
EXTRA_REPOS = {
    "deepseek-ai/DeepSeek-V4-Flash":            {"parameter_count": "168B", "quantization": "Q4_K_M"},
    "MiniMaxAI/MiniMax-M2.7":                   {"parameter_count": "228.7B", "quantization": "Q4_K_M"},
    "bullerwins/MiniMax-M2.7-REAP-172B-fp8":    {"parameter_count": "172B", "quantization": "FP8"},
    "cyankiwi/MiniMax-M2.7-AWQ-4bit":           {"parameter_count": "228.7B", "quantization": "AWQ-4bit"},
    **{repo: {} for repo in MLX_REPOS},
}

api = HfApi()


def main():
    with open(DATA_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    by_name = {m["name"]: m for m in catalog}
    existing = set(by_name)

    overwrite = "--overwrite" in sys.argv
    to_add = {}

    # Authors
    for author in AUTHORS:
        print(f"Fetching author: {author} ...", flush=True)
        models = list(api.list_models(author=author, full=True, cardData=True))
        print(f"  {len(models)} repos", flush=True)
        for mi in models:
            if mi.id in existing and not overwrite:
                continue
            ov = EXTRA_REPOS.get(mi.id)
            entry = _entry_from_modelinfo(mi, ov)
            if entry:
                to_add[mi.id] = entry

    # Explicit extra repos (not covered by an author scan)
    for repo, ov in EXTRA_REPOS.items():
        if repo in to_add:
            continue
        if repo in existing and not overwrite:
            continue
        try:
            mi = api.model_info(repo, files_metadata=False)
        except Exception as e:
            print(f"  SKIP {repo}: {e}", flush=True)
            continue
        entry = _entry_from_modelinfo(mi, ov)
        if entry:
            to_add[repo] = entry

    if not to_add:
        print("Nothing new to add.")
        return

    # Backup + merge
    with open(DATA_PATH + ".bak", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)
    for name, entry in to_add.items():
        by_name[name] = entry
    merged = list(by_name.values())
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"\nAdded/updated {len(to_add)} models. Catalog now {len(merged)} (was {len(catalog)}).")
    for n in sorted(to_add)[:20]:
        e = to_add[n]
        print(f"  + {n}  [{e['parameter_count']}, {e['quantization']}]")
    if len(to_add) > 20:
        print(f"  ... and {len(to_add) - 20} more")


if __name__ == "__main__":
    main()
