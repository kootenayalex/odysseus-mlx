# Cloud escalation (local-first, provider-agnostic)

The MLX gateway runs everything **local-first**. For genuinely hard tasks — long
context, complex reasoning — a serve can transparently route the request to a
**cloud** model and keep everything smaller on-box. The provider is a config
value, so switching vendors (or surviving a billing change like the Claude Code
MAX→API-key flip) is a one-line edit, not a code change.

This rides Rapid-MLX's built-in litellm cloud routing — see `_build_rapid_cmd`
in `routes/mlx_gateway_routes.py`. The serve command gains
`--cloud-model/--cloud-api-base/--cloud-threshold/--cloud-api-key` only when a
`data/mlx_autoserve.json` entry opts in.

## How escalation triggers
Rapid-MLX routes a request to the cloud only when it has **more new (uncached)
tokens than `cloud_threshold`** (default 20000). Small/normal requests stay
local. So escalation tracks *task size/hardness*, not a hard switch.

## The sensitivity gate (keep personal data local)
Cloud routing is **per-serve**. Put `cloud_model` ONLY on a serve meant for
non-sensitive work — e.g. a dedicated `escalate` (or `research`) model. Leave it
**off** the models email/utility use (`coder`, `chat`, `deepseek-coder` here), so
personal mail and routine agent work are never sent off-box.

Wire the escalate model to the roles that should be allowed to leave the house
(e.g. point the research role's model at `escalate`); email tasks resolve the
`utility`/`default` roles, which stay on the local-only models.

## Activate (drop in a key)
1. Add a dedicated entry to `data/mlx_autoserve.json` (NOT on an email model):

   ```json
   "escalate": {
     "repo_id": "Qwen/Qwen3-14B-MLX-4bit",
     "tool_call_parser": "qwen3",
     "cloud_model": "anthropic/claude-sonnet-4-5-20250929",
     "cloud_threshold": 20000,
     "cloud_api_key_env": "ANTHROPIC_API_KEY",
     "priority": 5
   }
   ```
   - `cloud_model` is a **litellm** string — any provider works
     (`anthropic/…`, `openai/gpt-4o`, `openrouter/…`, …).
   - For an OpenAI-compatible gateway/proxy, also set
     `"cloud_api_base": "https://your-endpoint/v1"`.
   - `cloud_api_key_env` names the env var holding the key — the key itself
     lives in `.env`, never in this file.

2. Put the key in `~/vault/dev/odysseus/.env`:
   ```
   ANTHROPIC_API_KEY=sk-...
   ```

3. Restart the gateway: `launchctl stop io.odysseus.server && launchctl start io.odysseus.server`.

## Swapping providers (the billing hedge)
Change `cloud_model` (and `cloud_api_base` if needed) + the key env var. One edit,
no code. Point it at Claude-API, a local proxy, OpenRouter, another vendor — or
remove `cloud_model` to go fully local again.

## Security note
When `cloud_api_key_env` is set and present, the key is passed as
`--cloud-api-key` on the serve command line (visible to local `ps` and the
mode-protected tmux serve log). That's acceptable on this single-user loopback
box. On a shared host, omit `cloud_api_key_env` and instead export the litellm
provider env var (e.g. `ANTHROPIC_API_KEY`) into the serve's environment so the
key never appears in a command line.
