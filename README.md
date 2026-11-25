<div align="center">

<h1>basalt-colab</h1>

<p><b>Uncensored DeepSeek-V4.1-Flash on Google Colab G4 (RTX PRO 6000 96 GB), built from first principles around Basalt v0.5.7.</b><br>
150–190+ tok/s solo · 159 tok/s 2-stream concurrent · up to 1M elastic FP8 KV · 32 GiB Disk Conversation Cache · Dynamic Vision.<br>
Zero-dependency Python 3.11+ CLI · Visual Web Control Console · OpenAI / Anthropic Gateway · DeepSeek Harness, Claude Code, Codex, OpenCode &amp; Cursor config generator.</p>

</div>

## Install & Quick Start

Requires **Python 3.11+**, an OpenSSH client (`ssh`, `ssh-keygen`), and a Google AI Pro (200 CU/month) or Colab Pro (100 CU/month) subscription. Zero third-party Python dependencies. **Cross-platform CI-verified on Linux, macOS, and Windows 10/11** (also works on WSL2, Docker/Remote SSH, and Android Termux).

```bash
# Install globally with uv (recommended)
uv tool install git+https://github.com/HyacinthLo/basalt-colab.git

# Or clone & install editable
git clone https://github.com/HyacinthLo/basalt-colab.git && cd basalt-colab
pip install --no-deps --no-build-isolation -e .
```

Start your Colab G4 endpoint (or open the Unified Control Console first without waking a GPU):

```bash
basalt-colab up            # sets up Colab CLI + OAuth on first run, boots G4 (~1.5-2 min)
basalt-colab up -d         # start and return immediately without blocking in watch view
basalt-colab console       # opens http://127.0.0.1:18080/_console (auto-authenticated)
basalt-colab connect dsh   # prints ready-to-paste DeepSeek Harness cordis.patch.yml snippet
```

```text
basalt-colab 0.3.0 · DeepSeek-V4.1-Flash-1017-UD-IQ1_M (Basalt v0.5.7)

Starting a Colab G4 GPU · ~1.5-2 minutes (prebuilt binary)

   0:11  ✓  G4 GPU allocated
   0:15  ✓  abliteration edit downloaded (43 KB)
   0:56  ✓  model backbone downloaded
   1:02  ✓  Basalt built / ready
   1:05  ✓  abliteration baked into weights (62/62)
   1:20  ✓  model loaded onto GPU + RAM
   1:20  ✓  backbone storage punched & reclaimed
   1:45  ✓  Engram table loaded
   1:47  ✓  warmup answer done

✓ Ready after 1:48

  Console    http://127.0.0.1:18080/_console
  Web UI     http://127.0.0.1:18080/
  Base URL   http://127.0.0.1:18080/v1
  API key    sk-ds41-...
  Model      DeepSeek-V4.1-Flash-1017-UD-IQ1_M (effort 50, uncensored)
  Engine     UD-IQ1_M · 262K×2 FP8 KV · vision dynamic · route balanced · ~160 tok/s

  Try it     basalt-colab ask "Hello!"
  Console    basalt-colab console
  Agents     basalt-colab connect [dsh|claude|codex|opencode|openai]
  Shell env  eval "$(basalt-colab env)"
  Stop       basalt-colab down
```

## Measured Performance (Colab G4 — RTX PRO 6000 Blackwell 96 GB)

| Metric | Measured (Basalt v0.5.7 + `UD-IQ1_M` / `UD-IQ2_XXS`) |
| --- | --- |
| VM cold start (prebuilt `sm_120` binary) | **~95–108s** (`build.sh` in **6s**, zero-GIL 128 MiB parallel download at **3.47 GB/s**) |
| Solo decode (`/v1/chat/completions`) | **148–160+ tok/s** (`UD-IQ1_M`) · **94–125 tok/s** (`UD-IQ2_XXS`, 98.3% VRAM hit, 0.3% PCIe) |
| OpenAI Responses (`/v1/responses`) | **193.8 tok/s** (DSpark speculative draft + 32 draft experts) |
| Anthropic Messages (`/v1/messages`) | **152.5 tok/s** (native Claude Code compatible) |
| 2-stream concurrent decode | **159.0 tok/s aggregate** (`78.7 + 80.3 tok/s`, 94.1% hit, 0.8% PCIe) |
| Context & KV cache | **Up to 1,048,576 (1M) per slot** via `slot-full-context` (FP8 KV: 0.97 GiB @ 1M; FP16 optional) |
| Disk Conversation Cache | **32 GiB** `/content/conv_cache` (99.8% KV restored in **0.5 ms** across rotated sessions) |
| Vision (`dynamic` mode) | **0 MiB idle VRAM / 0 MiB idle RAM**, ~1.0–1.6s on-demand image encode |
| Client-disconnect cancel (`MSG_PEEK`) | **0.48s** measured GPU decode abort after client hangup |
| Cost | **~8.9 CU/hr** while GPU runs · **0 CU** when stopped |

## Unified Control Console (`/_console`) & Basalt Web UI (`/`)

- **Control Console (`/_console`)**: Self-contained dark UI (zero CDN deps) featuring:
  - **One-Click Client Integration Tabs**: live-synced, copy-ready config snippets & shell env exports for **DeepSeek Harness (`dsh`)**, **Claude Code**, **Codex CLI**, **OpenCode**, and **Cursor / Cherry Studio / OpenAI** (`GET /_control/clients`).
  - **Single-Source `config.toml` + Quick Bar**: Quick controls for `Profile` (`Balanced` / `Solo-Fast` / `Exact`), `Quant`, `Effort`, and `Idle` that edit `config.toml` directly (never clobbering custom fields), with **Save & Apply** and **Save & Restart G4** buttons.
  - **Live Telemetry & Billing**: real-time Basalt `tok/s` + VRAM hit % + PCIe %, boot timeline, 14-day CU cost curve & session ledger (`usage.jsonl`), and local/VM log viewer.
- **Basalt Native Web UI (`/`)**: Once G4 is ready, `/` proxies 100% natively to Basalt's built-in chat & diagnostics console without rewriting any asset paths.
- Run `basalt-colab console` to open `/_console#key=...` auto-authenticated (the URL fragment is stored in `localStorage` and immediately stripped from the URL; never sent over HTTP).

## Commands

| Command | Description |
| --- | --- |
| `basalt-colab up [opts]` | Start G4 (`-d`/`--no-watch`, `--quant`, `--profile balanced\|solo-fast\|exact`, `--routing-mode`, `--parallel`, `--uncensor-scale`, `--wait-gpu [MIN]` default 120m / `0`=unlimited, `--no-notify`, `--basalt-ref`) |
| `basalt-colab restart [opts]` | Stop current G4, reload saved `config.toml`, and boot fresh (keeps local proxy running) |
| `basalt-colab down [--all]` | Stop G4, wipe VM secrets, record session in `usage.jsonl`, stop detached proxy |
| `basalt-colab console [--no-open]` | Ensure local proxy is up (without waking G4) and open `/_console` |
| `basalt-colab connect [CLIENT] [opts]` | Pure config generator for **`dsh`** (DeepSeek Harness), **`claude`**, **`codex`**, **`opencode`**, **`openai`** (`--raw`, `--json`, `--sh`, `--powershell`, `--write`) — synced to the live running model! |
| `basalt-colab env [--sh\|--powershell\|--json]` | Print live-synced `OPENAI_*`, `ANTHROPIC_*`, and `BASALT_API_KEY` exports |
| `basalt-colab ask [opts] TEXT` | One-shot stream (`--effort`, `--max`, `--no-think`, `--hide-think`, `--max-tokens`, `--system`, `--image`, or piped `stdin`) |
| `basalt-colab config [show\|path\|wizard\|get\|set\|profile]` | Inspect or update `config.toml` (`config set effort high`, `config profile solo-fast` — hot-reloads live!) |
| `basalt-colab status` / `watch` | Snapshot or live terminal bar (`tok/s`, VRAM hit %, PCIe %, idle countdown, CU cost) |
| `basalt-colab logs [-f\|--follow] [--local\|--vm]` | Local proxy log (`-f` live tail) + VM boot timeline + Basalt server & engine logs |
| `basalt-colab doctor` / `setup` | Environment diagnostics and one-time Colab OAuth + key setup |

## Quantization Tiers & Zero-SSD-Expert Staging

`basalt-colab` inspects the remote 64 KiB `.basalt` header before downloading and automatically plans storage so **0 routed experts spill to SSD**:

| Quant (`[basalt].quant`) | Pack Size | Storage Mode on G4 | Quality (vs BF16) | Notes |
| --- | --- | --- | --- | --- |
| **`ud-iq1_m`** *(default)* | 260 GB | **`ram+ram`** (100% RAM+VRAM) | KLD `0.2779` · Top-1 `83.88%` | Fastest (`150–190+ tok/s`), zero SSD I/O |
| **`ud-iq2_xxs`** | 285 GB | **`ram+hybrid`** (Experts 100% RAM+VRAM; **~78 GiB Engram in RAM** + **~25 GiB parallel SSD tail**) | KLD `0.2408` (**-13.4%**) · Top-1 `85.15%` | Official pack on `jesdga`, ~0 extra SSD write wait |
| **`ud-q2_k_xl`** *(experimental)* | ~335 GB | **`ssd+hybrid`** (`/content` punch-on-read backbone + ~36 GiB RAM Engram + ~67 GiB SSD tail) | Highest 2-bit tier | Build & upload with [`scripts/build_q2k_xl_pack.py`](scripts/build_q2k_xl_pack.py) |
| `NVFP4` *(unsupported)* | 428 GB | **Rejected fail-fast** | — | Backbone (`~294 GiB`) > G4 VRAM+RAM (`~272 GiB`); Basalt requires ≥320 GB RAM |

## Declarative Configuration (`config.toml`)

Stored at `~/.config/basalt-colab/config.toml` (`%APPDATA%\basalt-colab\config.toml` on Windows; override via `BASALT_COLAB_HOME`):

```toml
[colab]
session = "basalt-ds41"
idle_min = 30            # hot-reloadable live
max_hours = 6            # hot-reloadable live
min_balance = 5
wait_gpu = false         # hot-reloadable live
wait_gpu_min = 120       # hot-reloadable live (0 = unlimited when wait_gpu = true)
orphan_boot_s = 30

[proxy]
bind = "127.0.0.1"
port = 18080
strict_model = false     # hot-reloadable live (false = route claude-haiku-* / gpt-* -> active model)

[basalt]
quant = "ud-iq1_m"       # "ud-iq1_m", "ud-iq2_xxs", "ud-q2_k_xl", or "custom"
pack_url = ""            # optional URL override (required for ud-q2_k_xl / custom)
ref = ""                 # empty = pinned manifest (v0.5.7 + prebuilt fast path)
model_alias = ""         # empty = auto from quant (e.g. DeepSeek-V4.1-Flash-1017-UD-IQ1_M)
max_context = 262144     # per-slot up to 1048576 (1M) via native slot-full-context
parallel = 2             # 1..4 concurrent slots
kv = "fp8"               # "fp8" or "fp16"
vision = "dynamic"       # 0 MiB idle VRAM/RAM
spec = 6                 # DSpark speculative window (0..8)
draft_experts = 32       # 32, 64, 96, 128, or 0
routing_mode = "balanced" # "balanced" (+20-37% speed), "fast", or "exact" (lossless)
answer_room = "auto"     # "auto" reserves room for final answer so thinking never truncates it
conv_cache_mib = 32768   # 32 GiB disk conversation cache
resident_budget_gib = 0  # 0 = auto-planned per storage mode (55 / ~87 / ~130 GiB)
uncensor = true          # 43 KB mrfakename rank-2 stochastic-rounding abliteration (0% decode slowdown)
uncensor_scale = 1.0     # strength multiplier (0.0=off, 1.0=default KL 0.067, 1.15-1.25=stronger)
effort = "50"            # hot-reloadable live (1-100, low, high, max)

[basalt.sampling]
# temperature = 1.0
# top_p = 0.95
```

## Architecture & 6-Layer Cost Safety

See [docs/how-it-works.md](docs/how-it-works.md) and [SETUP.md](SETUP.md) for deep dives into 3-mode automatic storage staging, in-place rank-2 abliteration, the offline `UD-Q2_K_XL` pack converter (`scripts/build_q2k_xl_pack.py`), the GitHub Actions prebuilt `sm_120` release pipeline (`.github/workflows/build-basalt-prebuilt.yml`), and the 6-layer cost protection system.

## Credits

Built on [Basalt](https://github.com/jesdga95/basalt) by jesdga95, DeepSeek-V4.1-Flash by DeepSeek, Unsloth's Dynamic V3.1 quantizations (`UD-IQ1_M` / `UD-IQ2_XXS` / `UD-Q2_K_XL`), mrfakename's 43 KB rank-2 `abliteration-edit.npz`, Google's Colab CLI, and inspired by [leonxger/deepseek-colab](https://github.com/leonxger/deepseek-colab).
