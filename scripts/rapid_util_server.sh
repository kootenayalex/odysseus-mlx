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

# rapid-mlx >=0.9.x gates the audio/transcription lane behind an explicit
# opt-in on a text-mode boot (Task #292: enable_audio_lane OR
# is_audio_name(model_name)) -- our HOST_MODEL is a chat model, not an
# audio-named alias, so --enable-audio is required for /v1/audio/transcriptions
# (Whisper) to mount at all. Without it the server only serves $EMBED_MODEL.
exec "$RAPID_BIN" serve "$HOST_MODEL" \
  --served-model-name util --host 127.0.0.1 --port "$PORT" \
  --embedding-model "$EMBED_MODEL" --enable-audio \
  --no-thinking --log-level WARNING
