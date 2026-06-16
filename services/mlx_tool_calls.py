"""Tool-call lifting for MLX-served models — ported from Baton's gateway.

MLX backends (mlx_lm.server serving Qwen-family models) emit tool calls as
plain text, never as structured OpenAI `tool_calls`, and in several
inconsistent shapes seen in the wild:

  * Qwen3 (chat):  <tool_call>{"name": N, "arguments": {...}}</tool_call>
  * Qwen2.5-Coder: <tools>\\n {"name": N, "arguments": {...}}\\n</tools>
  * Qwen2.5-Coder: a bare {"name": N, "arguments": {...}} with no wrapper
  * Qwen2.5-Coder: a named tag  <N arguments='{...}'/>  (sometimes ```-fenced)

`extract_tool_calls()` handles all of these. When the set of offered tool
names is known we (a) also catch the named-tag form and (b) refuse to lift
any call whose name isn't an offered tool — which kills false positives on
example JSON/XML the model writes as illustration. `classify_stream()` lets
the streaming path forward prose while buffering a call until it can be
lifted.

Pure functions, no I/O — this is the crown jewel of the gateway and is fully
unit-tested in tests/test_mlx_tool_calls.py.
"""

from __future__ import annotations

import json
import re
import uuid

_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOLS_TAG_RE = re.compile(r"</?tools>")
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_MARKERS = ("<tool_call>", "<tools>")


def _try_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None


def tool_names(body: dict) -> list:
    """Function names offered in an OpenAI/Ollama request's `tools` array."""
    out = []
    for t in (body.get("tools") or []):
        fn = (t.get("function") or {}).get("name") if isinstance(t, dict) else None
        if fn:
            out.append(fn)
    return out


def _json_objects(text: str) -> list:
    """Source spans of every top-level {...} object in `text`, brace-matched and
    string-aware so braces inside string literals don't throw off the nesting."""
    objs, depth, start = [], 0, -1
    in_str = esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                objs.append(text[start:i + 1])
                start = -1
    return objs


def _named_tag_re(names):
    alt = "|".join(re.escape(n) for n in names)
    return re.compile(r"<(" + alt + r")\b[^>]*?arguments\s*=\s*(['\"])(.*?)\2[^>]*?>", re.DOTALL)


def _mk_tc(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name,
                         "arguments": args if isinstance(args, str) else json.dumps(args or {})}}


def extract_tool_calls(content: str, names=None):
    """Lift tool calls out of raw model text into OpenAI `tool_calls`. `names`,
    when given, is the set of offered tool names — used to catch the named-tag
    form and to reject calls to tools that weren't offered. Returns
    (tool_calls, cleaned_content); ([], content) when no call is present."""
    names = set(names or [])
    tcs, residue = [], content
    wrapped = _TOOLCALL_RE.findall(content)
    if wrapped:                                    # <tool_call>{...}</tool_call>
        residue = _TOOLCALL_RE.sub("", content)
        for raw in wrapped:
            d = _try_json(raw)
            if isinstance(d, dict) and "name" in d and (not names or d["name"] in names):
                tcs.append(_mk_tc(d["name"], d.get("arguments", d.get("parameters", {}))))
    if not tcs and names:                          # <name arguments='{...}'/>
        rx = _named_tag_re(names)
        hits = list(rx.finditer(content))
        if hits:
            residue = rx.sub("", content)
            for m in hits:
                tcs.append(_mk_tc(m.group(1), _try_json(m.group(3)) or m.group(3)))
    if not tcs:                                    # bare / <tools>-wrapped JSON
        scan = _THINK_RE.sub("", content)
        for o in _json_objects(scan):
            d = _try_json(o)
            if isinstance(d, dict) and "name" in d and (not names or d["name"] in names):
                tcs.append(_mk_tc(d["name"], d.get("arguments", d.get("parameters", {}))))
                residue = residue.replace(o, "")
        residue = _TOOLS_TAG_RE.sub("", residue)
    if not tcs:
        return [], content
    cleaned = _THINK_RE.sub("", residue).strip().strip("`").strip()
    cleaned = re.sub(r"^(xml|json)\b", "", cleaned).strip()   # leftover ```xml / ```json fence tag
    return tcs, (cleaned or None)


def _is_partial(s: str, marker: str) -> bool:
    """True if `s` is a nonempty strict prefix of `marker` (still being typed)."""
    return bool(s) and len(s) < len(marker) and marker.startswith(s)


def classify_stream(acc: str, names=()):
    """Decide how to treat assistant content accumulated so far in a stream.
    Returns (mode, safe_len): mode in {scan, prose, tool}; safe_len is how many
    leading chars are safe to forward as prose right now. A <think> block streams
    transparently; only what follows it decides prose-vs-tool. `names` lets us
    recognize a `<toolname ...>` opening tag as the start of a call."""
    s = acc.lstrip()
    off = len(acc) - len(s)
    if s.startswith("<think>"):
        end = acc.find("</think>")
        if end == -1:
            return "scan", len(acc)            # still thinking: all prose so far
        post = end + len("</think>")
    elif _is_partial(s, "<think>"):
        return "scan", off                     # might be opening <think>; hold it
    else:
        post = off
    rest = acc[post:].lstrip()
    if rest == "":
        return "scan", post                    # nothing after think yet
    # Coder sometimes wraps the call in a ```xml / ```json fence — peek past a
    # leading fence so the call inside is still recognized (and buffered).
    if rest.startswith("```"):
        nl = rest.find("\n")
        if nl == -1:
            return "scan", post                # still typing the ```lang line
        decide = rest[nl + 1:].lstrip()
        if decide == "":
            return "scan", post                # fence open, content not here yet
    else:
        decide = rest
    tags = list(_TOOL_MARKERS) + ["<" + n for n in names]
    if decide.startswith(tuple(tags)) or decide[:1] == "{":
        return "tool", post                    # a tool call starts here
    if any(_is_partial(decide, t) for t in tags):
        return "scan", post                    # might be a tool tag; hold it
    return "prose", len(acc)                    # ordinary text


def normalize_tool_calls(resp: dict, names=None) -> dict:
    """Lift model-emitted tool calls into OpenAI `tool_calls` (non-streaming)."""
    for ch in resp.get("choices", []):
        msg = ch.get("message") or {}
        if msg.get("tool_calls"):
            continue
        tcs, cleaned = extract_tool_calls(msg.get("content") or "", names)
        if tcs:
            msg["tool_calls"] = tcs
            msg["content"] = cleaned
            ch["finish_reason"] = "tool_calls"
    return resp


def ollama_messages_to_openai(messages: list) -> list:
    """Ollama-native messages -> OpenAI; base64 `images[]` -> `image_url` data URLs."""
    out = []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        imgs = m.get("images") or []
        if imgs:
            parts = [{"type": "text", "text": content}]
            for b64 in imgs:
                url = b64 if str(b64).startswith("data:") else f"data:image/png;base64,{b64}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
            out.append({"role": role, "content": parts})
        else:
            out.append({"role": role, "content": content})
    return out
