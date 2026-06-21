#!/usr/bin/env python3
"""Ensure the MLX Whisper repo has the HuggingFace processor files Rapid-MLX needs.

Rapid-MLX 0.8.0's mlx-audio Whisper loader calls `WhisperProcessor.from_pretrained(model_dir)`,
which needs `preprocessor_config.json` + tokenizer files. The `mlx-community/whisper-*-mlx`
weight repos ship only `config.json` + `weights.safetensors`, so transcription fails with
"Processor not found". This script copies the missing files (from a kept source dir, or by
re-fetching from the original openai repo) into the MLX repo's HF-cache snapshot.

Idempotent: a no-op once the files are present. Run before the utility server starts.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

MLX_REPO = os.environ.get("ODYSSEUS_MLX_WHISPER_REPO", "mlx-community/whisper-large-v3-turbo")
SRC_REPO = "openai/whisper-large-v3-turbo"  # original repo with the processor/tokenizer
KEPT_DIR = os.path.expanduser("~/.cache/odysseus-whisper/whisper-large-v3-turbo")
PROCESSOR_FILES = [
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt",
    "normalizer.json", "added_tokens.json", "generation_config.json",
]


def _snapshot_dir() -> str | None:
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    repo = "models--" + MLX_REPO.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(cache, repo, "snapshots", "*")))
    return snaps[-1] if snaps else None


def _source_file(fn: str) -> str | None:
    """A local copy of `fn`: prefer the kept dir, else fetch from the openai repo."""
    kept = os.path.join(KEPT_DIR, fn)
    if os.path.exists(kept):
        return kept
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(SRC_REPO, fn)
    except Exception:
        return None


def main() -> int:
    snap = _snapshot_dir()
    if not snap:
        print(f"[whisper-fixup] MLX repo {MLX_REPO} not in cache yet — skipping", file=sys.stderr)
        return 0
    added = []
    for fn in PROCESSOR_FILES:
        dst = os.path.join(snap, fn)
        if os.path.exists(dst):
            continue
        src = _source_file(fn)
        if src:
            shutil.copy(src, dst)
            added.append(fn)
    print(f"[whisper-fixup] {MLX_REPO}: {len(added)} file(s) added"
          f"{' (' + ', '.join(added) + ')' if added else ' — already complete'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
