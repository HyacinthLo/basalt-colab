# How `basalt-colab` Works

`basalt-colab` is engineered from first principles around **Basalt v0.5.7** and **DeepSeek-V4.1-Flash** (`UD-IQ1_M`, `UD-IQ2_XXS`, and experimental self-hosted `UD-Q2_K_XL`) on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM, ~176 GiB usable host RAM, ~220 GiB local SSD).

## 1. Adaptive 3-Mode Storage Pipeline (`basalt_colab/vm/boot.sh`)

Before downloading any weights, `basalt_colab/vm/basalt_jobs.py --plan` reads only the 64 KiB `BASALTMF v2` header via HTTP Range request and automatically chooses a zero-SSD-expert staging plan against the VM's real RAM, VRAM, and `/content` free space:

1. **`ram+ram` (`UD-IQ1_M`, 139.1 GiB backbone + 103.0 GiB Engram)**: Backbone downloads into `/mnt/ram/model.basalt`, shrinks to `~1.6 GiB` via punch-on-read during `POST /load`, then 100% of Engram streams into `/mnt/ram/engram.bin`. **100% RAM+VRAM, 0 SSD I/O.**
2. **`ram+hybrid` (`UD-IQ2_XXS`, 162.85 GiB backbone + 103.0 GiB Engram)**: Backbone downloads into `/mnt/ram/model.basalt` while **only the ~25 GiB Engram tail** streams to `/content/engram_tail.bin` **in parallel at `t = 2s`** (fully hidden behind backbone download & load, overcoming Colab `/content`'s ~290 MiB/s GCE PD write cap). After punch-on-read reclaims `/mnt/ram`, the remaining **~78 GiB Engram head** streams into `/mnt/ram/engram.bin` in ~25–30s. Patch #5 (`BASALT_COLAB_ENGRAM_SPLIT`) serves both files transparently through `DirectFile`. **0 cold experts on SSD, ~76% Engram in RAM!**
3. **`ssd+hybrid` (experimental self-hosted `UD-Q2_K_XL`, ~205 GiB backbone + 103.0 GiB Engram)**: Probes `FALLOC_FL_PUNCH_HOLE` on `/content` first, stages backbone on `/content/model.basalt`, punches ~200 GiB free during `POST /load`, then streams ~36 GiB Engram head into `/mnt/ram/engram.bin` + ~67 GiB tail into `/content/engram_tail.bin` concurrently.
4. **Fail-fast guard**: Any pack whose backbone exceeds `VRAM + RAM` zero-SSD-expert capacity (e.g. `NVFP4`, ~294 GiB backbone) is rejected in 1 second before spending CU.

Boot sequence (`~90–95s` with GitHub Actions prebuilt `sm_120` tarball):
1. **Parallel**: Multi-process native C `curl | dd` (`pfetch.py`, 128 MiB RAM chunks, ~3.45+ GB/s zero-GIL) streams the sparse backbone, 43 KB rank-2 `abliteration-edit.npz` (`~1s`), `vm/build.sh` (fetches prebuilt `basalt` + `abliterate` tarball and installs missing Python deps into `/mnt/ram/pydeps` in ~4s, avoiding `/content` `loop0` write contention), and (in `ram+hybrid`) background SSD download of the ~25 GiB Engram tail.
2. **In-place Abliteration (`~1.5–3s`)**: `abliterate_basalt.py --scale $UNCENSOR_SCALE` + standalone OpenMP `abliterate.c` applies mrfakename's 43 KB rank-2 edit directly across all 62/62 target `Q8_0`/`Q5_K`/`Q6_K` tensors via stochastic rounding (0% decode slowdown).
3. **Load & Punch-on-Read (`~10–15s`)**: `POST /load` moves hot experts to VRAM (~78 GiB) and resident complement to 2 MiB `MADV_HUGEPAGE` pinned RAM (parallel `cudaHostRegister`). Patch #3 (`BASALT_PUNCH_FILE`) calls `fallocate(FALLOC_FL_PUNCH_HOLE)` after VRAM and RAM waves, shrinking `/mnt/ram/model.basalt` down to **~1.6 GiB**.
4. **Engram Stream (`~24s`)**: Streams the RAM Engram (`103 GiB` in `ram+ram`, `~78 GiB` in `ram+hybrid`) into `/mnt/ram/engram.bin` at ~3.4+ GB/s (served via zero-copy `mmap()`) and joins the background SSD tail (already finished).
5. **Warmup & Gate (`~1s`)**: Verifies generation, opens `gate.py` on `:18080`, and arms the VM-side hard-cap backstop (`MAX_HOURS * 3600 + 300s`).

## 2. Native Basalt v2 Configuration (`basalt_colab/vm/serve.sh`)

- **`slot-full-context: true`**: Each slot receives the full `basalt.max_context` (up to `1,048,576` / 1M tokens) without `llama.cpp`-style `parallel × context` multiplication.
- **Performance Profiles & Routing Modes**: Default `balanced` (`parallel=2`, `vram-reserve=450 MiB`, `--route-prior 0.1 --route-tail-skip 5`, KLD `+0.0005` within placement noise, +20–37% decode speed & longer speculative windows); optional `solo-fast` (`parallel=1`, `route-tail-skip=3`, `vram-reserve=400 MiB`) and `exact` (100% lossless routing).
- **Elastic FP8/FP16 KV**: Default `fp8` uses 920 B/token (`0.24 GiB` idle, `0.97 GiB` at 1M); `fp16` keeps unrounded KV (`3,200 B/token`).
- **Dynamic Vision**: `vision = "dynamic"` consumes **0 MiB idle VRAM and 0 MiB idle RAM** (identical memory plan to `vision = "off"`).
- **32 GiB Disk Conversation Cache**: `/content/conv_cache` snapshots evicted KV slots as `.bcs` files and restores them in `<1 ms` (99.8% hit verified across rotated multi-turn sessions).
- **8 Safe Patches (`patch_basalt.py`)**: `BASALT_COLAB_ENGRAM_ENV`, `BASALT_COLAB_TMPFS_FALLBACK`, `BASALT_COLAB_ENGRAM_SPLIT`, `BASALT_COLAB_PUNCH_ON_READ`, `BASALT_COLAB_PUNCH_GUARD`, `BASALT_COLAB_ENGRAM_MMAP` (zero-copy `mmap()` RAM head), `BASALT_COLAB_ENGRAM_TAIL_CACHE` (`1024 MiB` tail-only `RowCache`), and `BASALT_COLAB_RAM_HUGEPAGE_PIN` (`MADV_HUGEPAGE` + parallel `cudaHostRegister`).

## 3. Enhanced Local Gateway & 6-Layer Cost Protection

- **Unified Control Console (`/_console`) + Native Basalt Web UI (`/`)**: `/` proxies untouched to Basalt's built-in console once G4 is ready, and shows `/_console` while G4 is off/starting. `/_console` is authenticated with `api_key` (auto-loaded from `#key=` fragment by `basalt-colab console`), renders live Basalt telemetry, plots 14-day CU billing curves from `usage.jsonl`, and hot-reloads `config.toml` online.
- **6-Layer Cost Safety**:
  1. `0.5s` `MSG_PEEK` client-disconnect cancellation (aborts active GPU Prefill/Decode AND unwakes cold-start waiters).
  2. Orphan cold-start auto-cancel (`orphan_boot_s`, default `30s`).
  3. Idle auto-stop (`idle_min`, default `10 min`, hot-reloadable live).
  4. Proxy hard cap (`max_hours`, default `6 h`, hot-reloadable live).
  5. Detached local `_watchdog` process (survives proxy SIGKILL).
  6. VM-side hard-cap timer in `boot.sh` (kills Basalt and wipes `/content/secrets` if local machine sleeps).
