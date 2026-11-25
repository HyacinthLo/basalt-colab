# Setup & Reference (`basalt-colab`)

## Installation

Requires **Python 3.11+**, OpenSSH (`ssh`, `ssh-keygen`), and a Google AI Pro or Colab Pro subscription.

```bash
# Recommended: install as an isolated CLI tool with uv
uv tool install git+https://github.com/HyacinthLo/basalt-colab.git

# Or from a local clone
git clone https://github.com/HyacinthLo/basalt-colab.git && cd basalt-colab
pip install --no-deps --no-build-isolation -e .
```

Run `basalt-colab doctor` at any time to verify Python, OpenSSH, Colab CLI, OAuth balance, `config.toml`, VM payload integrity, and port availability.

## Paths & Legacy Migration

- **Config**: `~/.config/basalt-colab/config.toml` (`%APPDATA%\basalt-colab\config.toml` on Windows)
- **State**: `~/.local/state/basalt-colab/` (`api_key`, `id_ed25519`, `session.json`, `serve.lock`, `usage.jsonl`, `opencode.json`, `basalt-colab.log`)
- **Override**: Set `BASALT_COLAB_HOME=/path/to/dir` to keep config and state together in one custom folder.
- **Legacy migration**: On first `basalt-colab setup` / `up`, if `./state/api_key` or `./state/id_ed25519` exists from an older `deepseek-colab` checkout, it is safely copied (`0600`) into the new state directory.

## Control Console & Clients

- **Unified Control Console**: `basalt-colab console` starts the local proxy (without waking G4) and opens `http://127.0.0.1:18080/_console#key=...`. From there you can start/restart/stop G4, copy live-synced configs for all supported agents, edit & hot-reload `config.toml` via quick controls or raw TOML (with **Save & Apply** or **Save & Restart G4**), inspect Basalt telemetry & 14-day CU cost curve, and tail local/VM logs.
- **Basalt Web UI**: `http://127.0.0.1:18080/` proxies natively to Basalt's web console once G4 is ready.
- **Pure Config Generation (`basalt-colab connect`)**: All agents use clean config generation synced to the live running model:
  - **DeepSeek Harness (`dsh`)**: `basalt-colab connect dsh` (or `--raw`) outputs a ready-to-paste `llm-pi-ai` block for `~/.dsh/profiles/web/cordis.patch.yml` with DeepSeek-V4.1 efforts (`off: none`, `low`, `high`, `max`), `compat` switches, and Dynamic Vision (`input: [text, image]`).
  - **Claude Code**: `basalt-colab connect claude` prints `~/.claude/settings.json` and `ANTHROPIC_*` shell exports.
  - **Codex CLI**: `basalt-colab connect codex` prints `~/.codex/config.toml` (`wire_api = "responses"`) and `OPENAI_*` exports.
  - **OpenCode**: `basalt-colab connect opencode` prints `opencode.json` using `{env:OPENAI_API_KEY}` (pass `--write` to update the managed file for `OPENCODE_CONFIG=...`).
  - **Shell env**: `eval "$(basalt-colab env)"` exports `OPENAI_*`, `ANTHROPIC_*`, and `BASALT_API_KEY` synced to the running proxy.
- **Multimodal & Piped CLI**:
  ```bash
  basalt-colab ask --image screenshot.png "Explain this stack trace"
  git diff | basalt-colab ask --max "Review this diff for subtle bugs"
  ```

## Prebuilt Basalt Binary & `--basalt-ref`

- By default (`basalt.ref = ""`), `vm/build.sh` checks the pinned prebuilt release archive from `basalt_colab/vm/manifest.json` (`prebuilt_url` + `prebuilt_sha256`), skipping the ~85s C++ compile and bringing VM boot down to **~95s**. If unavailable or hash-mismatched, it transparently falls back to building from source on G4.
- To test an upstream Basalt branch, tag, or commit:
  ```bash
  basalt-colab up --basalt-ref main
  ```
  This skips the prebuilt cache, clones that ref, applies `patch_basalt.py` (with exact file/anchor mismatch diagnostics if upstream moved code), and compiles on G4.

## Development & Offline Tests

```bash
python3 -B -m unittest discover -s tests -v
```
