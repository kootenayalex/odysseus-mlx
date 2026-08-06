#!/usr/bin/env python3
"""Health watchdog for the Odysseus MLX serving stack.

launchd's KeepAlive restarts a *crashed* process, but not one that's alive yet
unresponsive (a hung serve, a wedged event loop). This probes the two standing
HTTP services and recovers a hung one — guarded by a failure threshold and a
per-service cooldown so it can never restart-loop.

Run periodically by launchd (StartInterval). One probe cycle per invocation;
state (consecutive failures, last restart) persists in data/watchdog_state.json.

Targets (each: launchd label + health URL):
  - io.odysseus.server   → http://127.0.0.1:7860/api/health   (gateway/app)
  - io.odysseus.rapid-util → http://127.0.0.1:8133/v1/models  (embeddings + STT)

Tunables (env):
  ODYSSEUS_WD_THRESHOLD   consecutive failures before restart   (default 3)
  ODYSSEUS_WD_COOLDOWN_S  min seconds between restarts/service   (default 600)
  ODYSSEUS_WD_TIMEOUT_S   per-probe HTTP timeout                 (default 6)
  ODYSSEUS_WD_PUSH_URL    optional Uptime-Kuma push base; on all-healthy we GET
                          "<url>?status=up&msg=ok" (heartbeat). Empty = off.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE_PATH = os.path.join(ROOT, "data", "watchdog_state.json")
LOG_PATH = os.path.join(ROOT, "logs", "watchdog.log")

THRESHOLD = int(os.environ.get("ODYSSEUS_WD_THRESHOLD", "3"))
COOLDOWN_S = int(os.environ.get("ODYSSEUS_WD_COOLDOWN_S", "600"))
TIMEOUT_S = float(os.environ.get("ODYSSEUS_WD_TIMEOUT_S", "6"))
PUSH_URL = os.environ.get("ODYSSEUS_WD_PUSH_URL", "").strip()

# Both original targets are retired: io.odysseus.server (:7860 gateway) on
# 2026-08-01 in favour of LM Studio :1234, and io.odysseus.rapid-util (:8133)
# on 2026-08-06 — dictation moved to the on-demand whisper.cpp sidecar (:8786)
# and embeddings to LM Studio. Nothing left to watch; re-enabling the watchdog
# with these labels would restart-loop services that no longer exist.
TARGETS: list[dict[str, str]] = []


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line)


def load_state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return {}


def save_state(state: dict) -> None:
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f)
    except OSError:
        pass


def probe(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def restart(label: str) -> bool:
    """Hard-restart a launchd service (kill + relaunch). KeepAlive brings it back
    even if kickstart is unavailable."""
    uid = os.getuid()
    try:
        subprocess.run(["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
                       capture_output=True, timeout=30)
        return True
    except Exception as e:
        log(f"  restart of {label} failed: {e}")
        return False


def main() -> int:
    state = load_state()
    now = time.time()
    all_ok = True

    for t in TARGETS:
        label, url = t["label"], t["url"]
        st = state.setdefault(label, {"fails": 0, "last_restart": 0})
        if probe(url):
            if st["fails"]:
                log(f"{label}: recovered (was {st['fails']} fail(s))")
            st["fails"] = 0
            continue

        all_ok = False
        st["fails"] += 1
        log(f"{label}: probe FAILED ({st['fails']}/{THRESHOLD}) {url}")
        if st["fails"] >= THRESHOLD:
            since = now - st.get("last_restart", 0)
            if since < COOLDOWN_S:
                log(f"  {label}: threshold hit but in cooldown ({int(COOLDOWN_S - since)}s left) — not restarting")
            else:
                log(f"  {label}: restarting (launchctl kickstart -k)")
                if restart(label):
                    st["last_restart"] = now
                    st["fails"] = 0

    save_state(state)
    if all_ok and PUSH_URL:
        try:
            urllib.request.urlopen(f"{PUSH_URL}?status=up&msg=ok", timeout=TIMEOUT_S).read()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
