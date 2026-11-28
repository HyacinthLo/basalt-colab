# AGENT_START_HERE (`basalt-colab`)

`basalt-colab` is a zero-dependency Python 3.11+ package (`dependencies = []`, Linux/macOS/Windows) that runs uncensored **DeepSeek-V4.1-Flash (`UD-IQ1_M` / `UD-IQ2_XXS` / `UD-Q2_K_XL` via `--pack-url`)** on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB) via **Basalt v0.6.0**, behind a local OpenAI + Anthropic + Unified Control Console gateway.

```text
clients / browser --HTTP--> basalt-colab :18080 (/_console, /_control/*, /v1/*, /)
                              └── SSH tunnel --> G4 VM: gate.py :18080 --> basalt serve :18081
```

## Key Commands

| Command | Effect | Costs CU? |
| --- | --- | --- |
| `basalt-colab doctor` | verifies Python 3.11+, OpenSSH, Colab CLI, OAuth balance, `config.toml`, VM payload & port | no |
| `basalt-colab setup` | installs Colab CLI, runs Google OAuth, generates keys & `config.toml`, migrates legacy `./state/` | no |
| `basalt-colab console [--no-open]` | starts local proxy (no GPU) and opens `/_console#key=...` | no |
| `basalt-colab up [--profile exact\|solo\|balanced] [--quant Q] [--wait-gpu [MIN]]` | allocates G4, boots Basalt (~1.5–2 min, default lossless `exact` routing & `effort=high`), prints endpoints | **yes (~8.9 CU/hr)** |
| `basalt-colab restart [up opts]` | stops current G4, reloads `config.toml`, and boots fresh (keeps local proxy running) | **yes (~8.9 CU/hr)** |
| `basalt-colab status` / `watch` | endpoints, Basalt live telemetry (SSE push in `watch`), CU billing ledger, Colab balance | no |
| `basalt-colab config [show\|path\|wizard\|get\|set\|profile]` | inspect or edit `config.toml` (hot-reloads `HOT_FIELDS` live via `/_control/reload`) | no |
| `basalt-colab connect [dsh\|claude\|codex\|opencode\|openai]` / `env` | generates live-synced client config snippets or shell/PowerShell env exports | no |
| `basalt-colab ask [--max] [--effort V] [--image P] TEXT` | streams answer with multimodal image + piped stdin support | while GPU runs |
| `basalt-colab logs [-f] [--local\|--vm]` | tails local proxy log or fetches VM boot timeline + Basalt server/engine logs | no |
| `basalt-colab down [--all]` | stops G4, wipes `/content/secrets`, writes `usage.jsonl` entry, stops proxy | no |

## Agent Rules

1. **Always ask before starting a G4** (`up` / `restart` / `serve` / `/_control/up`) — G4 bills ~8.9 CU/hr.
2. **Never stop a user's running G4 without explicit instruction**, and always confirm `not running` after stopping when asked.
3. **Never leak `api_key`, OAuth tokens, or `HF_TOKEN`.**
4. **Never invoke `colab ssh --proxy-mode` for a session `colab sessions` does not list** — it silently creates a paid VM.
5. **Keep runtime zero-dependency (`dependencies = []`) and cross-platform** (Linux/macOS/Windows); verify with `ruff check basalt_colab`, `mypy basalt_colab`, and `python3 -m unittest discover -s tests -v`.

## Package Map

- [`basalt_colab/paths.py`](basalt_colab/paths.py): XDG config/state resolution (`BASALT_COLAB_HOME`), cross-platform OS kernel lock (`serve.lck`), atomic writes, and legacy `state/` migration.
- [`basalt_colab/manifest.py`](basalt_colab/manifest.py) + [`basalt_colab/vm/manifest.json`](basalt_colab/vm/manifest.json): pinned Basalt tag/commit, prebuilt release archive URL/SHA-256, quant pack catalog, abliteration edit URL/SHA-256.
- [`basalt_colab/config.py`](basalt_colab/config.py): typed `config.toml` (`tomllib` + TOML writer), `BootOverrides`, `HOT_FIELDS` single source of truth, profiles (`exact`, `solo`, opt-in lossy `balanced`), validation, and `config_meta()`.
- [`basalt_colab/clients.py`](basalt_colab/clients.py): live-synced config & env generator for `dsh`, `claude`, `codex`, `opencode`, and `openai`.
- [`basalt_colab/usage.py`](basalt_colab/usage.py): append-only `usage.jsonl` CU billing ledger & daily/monthly aggregation.
- [`basalt_colab/colab.py`](basalt_colab/colab.py) + [`basalt_colab/keyring_shim/basalt_colab_keyring.py`](basalt_colab/keyring_shim/basalt_colab_keyring.py): Colab CLI wrapper, Linux headless `0600` keyring fallback, `vmx()`, `vm_logs()`, SSH builder, lockfile, LF-normalized tarball bundler, redacting logger.
- [`basalt_colab/lifecycle.py`](basalt_colab/lifecycle.py): `VmState` (`StrEnum`) & `Manager` (backoff alloc, orphan-boot cancel, tunnel repair, SSE event pub/sub, cached Basalt telemetry, watchdog).
- [`basalt_colab/proxy.py`](basalt_colab/proxy.py) + [`basalt_colab/console.py`](basalt_colab/console.py) + [`basalt_colab/web/console.html`](basalt_colab/web/console.html): HTTP proxy, CORS, `rewrite_request`, zero-polling `selectors` client-disconnect cancel, SSE `/_control/events`, `/_control/*`, and static Control Console asset.
- [`basalt_colab/term.py`](basalt_colab/term.py) + [`basalt_colab/cli.py`](basalt_colab/cli.py): terminal UI helpers and strict `argparse` CLI entrypoint (`basalt-colab` / `python3 -m basalt_colab`).
- [`basalt_colab/vm/`](basalt_colab/vm/): packaged VM scripts (`boot.sh` with Stage 0 NVMe probe & Stage 1 `wave+hybrid`, `wave_fetch.sh`, `build.sh`, `serve.sh`, `patch_basalt.py` with Stage 1 Micro-Wave loader + multi-window lazy Engram + `RowCache` filter + `INFLIGHT=256`, `basalt_jobs.py`, `abliterate_basalt.py`, `abliterate.c`, `pfetch.py`, `gate.py`).
- [`docs/how-it-works.md`](docs/how-it-works.md) & [`docs/q2k-xl-acceleration.md`](docs/q2k-xl-acceleration.md): architecture overview and `UD-Q2_K_XL` acceleration blueprint (Basalt `v0.6.0` + Stage 0 + Stage 1 Micro-Wave + Engram landed; optional Stage 2 & official HF pack pending).

## Upstream Version Governance

Always verify upstream releases (`jesdga95/basalt`, `googlecolab/google-colab-cli`) before bumping pins, maintain the verified list in [`basalt_colab/vm/manifest.json`](basalt_colab/vm/manifest.json), and point `colab_cli` at the **latest verified version** so fresh installs receive the newest verified release while `basalt-colab doctor` flags unverified installs.

- **`google-colab-cli` verified matrix** (last audited `2026-11-27`):
  - **`0.8.0`** — **Verified & pinned default** (`colab_cli`). Fixes Windows SSH portability (`#181`), `exec --timeout` busy-wait (`#185`), `0600` state/log permissions (`#196`), and piped `ssh` auto-update suppression. Uses native Keychain / Windows Credential Manager on macOS & Windows; on Linux, `colab_env()` injects the zero-dep [`basalt_colab/keyring_shim/basalt_colab_keyring.py`](basalt_colab/keyring_shim/basalt_colab_keyring.py) (`FallbackFileKeyring`) which delegates to native GNOME Keyring / KWallet when available and falls back to an owner-only (`0600`) atomic file store on headless Linux/WSL2/Docker/SSH/Termux (with automatic migration from `0.7.5` `token.json`).
  - `0.7.5`, `0.7.6` — **Verified & supported** (`colab_cli_supported`). Existing installs continue to work; `basalt-colab doctor` notes `0.8.0` as the latest verified release.
  - `1.0.0` *(upcoming, upstream `#289`)* — **Watch closely**: migrates to Colab Public API + GCP quota project (`colab init-gcp-project`); requires explicit setup/OAuth adaptation & cross-platform verification before upgrading.
