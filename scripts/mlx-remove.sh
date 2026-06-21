#!/usr/bin/env bash
#
# mlx-remove.sh — safely remove an MLX model from the odysseus gateway.
#
# Removing weights from the HF cache alone is unsafe: the gateway auto-downloads
# with no fallback, so any autoserve alias / settings entry still pointing at the
# model silently re-pulls it on the next request. This wrapper cleans the
# references FIRST, in the right order:
#
#   1. refuse if the model is (or is the alias behind) settings.default_model
#   2. remove any data/mlx_autoserve.json alias whose repo_id is the model
#   3. clear any other active model-setting that resolves to it (--force)
#   4. drop the model + its removed aliases from model_endpoints.cached_models
#   5. delete ~/.cache/huggingface/hub/models--<repo>
#
# Config is backed up to /tmp/mlx-remove-bak/ before any change.
#
# Usage:
#   scripts/mlx-remove.sh [-n|--dry-run] [-f|--force] <repo_id | autoserve_alias>
#
#   <repo_id>   e.g. mlx-community/Qwen3-8B-4bit   (an autoserve alias like
#               'chat' is accepted and resolved to its repo_id)
#   -n          preview the plan, change nothing
#   -f          proceed even when an autoserve alias / non-default setting
#               points at the model (those references are removed)
#
set -euo pipefail

usage() { sed -n '3,24p' "$0" | sed 's/^# \{0,1\}//'; }

DRY=0; FORCE=0; TARGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=1 ;;
    -f|--force)   FORCE=1 ;;
    -h|--help)    usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown flag: $1" >&2; usage >&2; exit 2 ;;
    *)  if [ -z "$TARGET" ]; then TARGET="$1"; else echo "too many args" >&2; exit 2; fi ;;
  esac
  shift
done
[ -n "${TARGET:-}" ] || { usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
AUTOSERVE="$ROOT/data/mlx_autoserve.json"
SETTINGS="$ROOT/data/settings.json"
APPDB="$ROOT/data/app.db"
HUB="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface/hub}}"

DRY="$DRY" FORCE="$FORCE" python3 - "$TARGET" "$AUTOSERVE" "$SETTINGS" "$APPDB" "$HUB" <<'PYEOF'
import json, os, sys, shutil, sqlite3, time

arg, autoserve_p, settings_p, appdb_p, hub = sys.argv[1:6]
DRY   = os.environ.get("DRY")   == "1"
FORCE = os.environ.get("FORCE") == "1"

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save(p, obj):
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)            # atomic

def entry_repo(v):
    if isinstance(v, str):  return v
    if isinstance(v, dict): return v.get("repo_id")
    return None

autoserve = load(autoserve_p)
settings  = load(settings_p)

# accept an autoserve alias name and resolve it to the real repo_id
repo_id = arg
if arg in autoserve and entry_repo(autoserve[arg]):
    repo_id = entry_repo(autoserve[arg])
    print(f"note: '{arg}' is an autoserve alias -> {repo_id}")

cache_name = "models--" + repo_id.replace("/", "--")
cache_dir  = os.path.join(hub, cache_name)
aliases_hit = sorted(n for n, v in autoserve.items() if entry_repo(v) == repo_id)

def resolves_to(val):
    if not val:
        return False
    if val == repo_id:
        return True
    v = autoserve.get(val)
    return v is not None and entry_repo(v) == repo_id

MODEL_KEYS = ("default_model", "chat_model", "coder_model", "task_model",
              "teacher_model", "research_model", "utility_model", "vision_model")
setting_hits = [(k, settings.get(k)) for k in MODEL_KEYS if resolves_to(settings.get(k))]

# ---- safety gates ----
if any(k == "default_model" for k, _ in setting_hits):
    print(f"\nREFUSED: default_model='{settings.get('default_model')}' resolves to {repo_id}.")
    print("Change default_model first (manage_settings, or edit data/settings.json), then re-run.")
    sys.exit(3)

other_hits = [(k, v) for k, v in setting_hits if k != "default_model"]
if other_hits and not FORCE:
    print(f"\nREFUSED: active model settings resolve to {repo_id}:")
    for k, v in other_hits:
        print(f"  {k} = {v}")
    print("Repoint them, or re-run with --force to clear them.")
    sys.exit(3)

if aliases_hit and not FORCE:
    print(f"\nREFUSED: autoserve alias(es) point at {repo_id}: {', '.join(aliases_hit)}")
    print("They would 404 after deletion. Repoint them in data/mlx_autoserve.json,")
    print("or re-run with --force to delete those alias entries.")
    sys.exit(3)

# ---- plan ----
def dir_size(p):
    # HF cache keeps real bytes in blobs/ and snapshots/ are symlinks to them,
    # so skip symlinks to avoid double-counting.
    total = 0
    for root, _, files in os.walk(p):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total

size = dir_size(cache_dir) if os.path.isdir(cache_dir) else 0
print(f"\nRemoving MLX model: {repo_id}")
print(f"  cache dir : {cache_dir} ({size/1e9:.1f} GB)" if os.path.isdir(cache_dir)
      else f"  cache dir : {cache_dir} (MISSING — refs cleanup only)")
print(f"  aliases   : {', '.join(aliases_hit) if aliases_hit else '(none)'}")
if other_hits:
    print(f"  settings  : will clear {', '.join(k for k, _ in other_hits)}")

remove_tokens = {repo_id} | set(aliases_hit)

if DRY:
    print("\n[dry-run] no changes made.")
    sys.exit(0)

# ---- execute ----
ts  = time.strftime("%Y%m%d-%H%M%S")
bak = "/tmp/mlx-remove-bak"
os.makedirs(bak, exist_ok=True)

if aliases_hit:
    shutil.copy(autoserve_p, f"{bak}/mlx_autoserve.json.{ts}")
    for a in aliases_hit:
        autoserve.pop(a, None)
    save(autoserve_p, autoserve)
    print(f"  ok autoserve: removed {', '.join(aliases_hit)}")

if other_hits:
    shutil.copy(settings_p, f"{bak}/settings.json.{ts}")
    for k, _ in other_hits:
        settings[k] = ""
    save(settings_p, settings)
    print(f"  ok settings: cleared {', '.join(k for k, _ in other_hits)}")

if os.path.exists(appdb_p):
    con = sqlite3.connect(appdb_p)
    cur = con.cursor()
    try:
        rows = cur.execute("SELECT id, cached_models FROM model_endpoints").fetchall()
    except sqlite3.Error as e:
        rows = []
        print(f"  ! app.db: {e}")
    if rows:
        with open(f"{bak}/cached_models.{ts}.txt", "w") as f:
            for rid, cm in rows:
                f.write(f"{rid}|{cm}\n")
    changed = 0
    for rid, cm in rows:
        if not cm:
            continue
        try:
            lst = json.loads(cm)
        except Exception:
            continue
        new = [x for x in lst if x not in remove_tokens]
        if new != lst:
            cur.execute("UPDATE model_endpoints SET cached_models=? WHERE id=?",
                        (json.dumps(new), rid))
            changed += 1
    con.commit()
    con.close()
    print(f"  ok app.db: cleaned cached_models in {changed} endpoint row(s)")

if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)
    print(f"  ok weights: deleted ({size/1e9:.1f} GB freed)")
else:
    print("  -  weights already absent")

print(f"\nDone. Config backups in {bak}/ (*.{ts})")
PYEOF
