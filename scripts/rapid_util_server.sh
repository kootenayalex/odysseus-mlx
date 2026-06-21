#!/bin/bash
# Standing Rapid-MLX "utility" server: hosts a small chat model (required by
# `rapid-mlx serve`) plus the embeddings endpoint (all-MiniLM — the SAME model
# the legacy mlx-openai-server embed service used, so the RAG vector store stays
# valid) and Whisper STT on demand. Replaces io.odysseus.mlx-embed AND the
# scheduler-managed whisper serve in one process. Port 8133 (unchanged from the
# old embed service so EMBEDDING_URL host:port is stable).
set -euo pipefail

RAPID_BIN="${ODYSSEUS_RAPID_MLX_BIN:-$HOME/.local/bin/rapid-mlx}"
RAPID_PY="$HOME/.local/share/uv/tools/rapid-mlx/bin/python"
HOST_MODEL="${ODYSSEUS_RAPID_UTIL_HOST:-mlx-community/Phi-3.5-mini-instruct-4bit}"
EMBED_MODEL="${EMBEDDING_MODEL:-mlx-community/all-MiniLM-L6-v2-4bit}"
PORT="${ODYSSEUS_RAPID_UTIL_PORT:-8133}"

# Ensure the Whisper repo has the processor files mlx-audio needs (idempotent).
"$RAPID_PY" "$(dirname "$0")/rapid_whisper_fixup.py" || true

exec "$RAPID_BIN" serve "$HOST_MODEL" \
  --served-model-name util --host 127.0.0.1 --port "$PORT" \
  --embedding-model "$EMBED_MODEL" \
  --no-thinking --log-level WARNING
