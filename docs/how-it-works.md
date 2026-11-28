# How `basalt-colab` Works

`basalt-colab` is engineered from first principles around **Basalt v0.6.0** and **DeepSeek-V4.1-Flash** (`UD-IQ1_M`, `UD-IQ2_XXS`, and `UD-Q2_K_XL`) on a single Google Colab G4 (RTX PRO 6000 Blackwell 96 GB VRAM, ~176 GiB usable host RAM, ~220 GiB local SSD).

## 1. Stage 0 Scratch Probe & Adaptive Storage Pipeline (`basalt_colab/vm/boot.sh`)

Before planning storage, `boot.sh` runs a **Stage 0** probe over `/proc/mounts` for any writable local `ext4`/`xfs`/`btrfs` volume with `>= 120 GiB` free and `> 600 MiB/s` synced write throughput (setting `SSD_DIR` + `CONV_CACHE_DIR`; falling back cleanly to `/content` if absent).

Then `basalt_colab/vm/basalt_jobs.py --plan` reads the 64 KiB `BASALTMF v2` header (plus the small `meta` JSON for hybrid packs) via HTTP Range and selects a zero-SSD-expert staging plan against the VM's real RAM, VRAM, and `$SSD_DIR` free space:

1. **`ram+ram` (`UD-IQ1_M`, 139.1 GiB backbone + 103.0 GiB Engram)**: Backbone downloads into `/mnt/ram/model.basalt`, shrinks to `~1.6 GiB` via punch-on-read during `POST /load`, then 100% of Engram streams into `/mnt/ram/engram.bin` (zero-copy `mmap()`, `RowCache` disabled). **100% RAM+VRAM, 0 SSD I/O.**
2. **`ram+hybrid` (`UD-IQ2_XXS`, 162.85 GiB backbone + 103.0 GiB Engram)**:
   - Backbone downloads into `/mnt/ram/model.basalt` while the logical full-size sparse SSD file (`$SSD_DIR/engram_tail.bin`) downloads the predicted SSD subset **in parallel at `t = 2s`** (hidden behind backbone download & load).
   - During `POST /load`, `BASALT_ENGRAM_LAZY_OPEN=1` allocates the **`4 GiB` `RowCache`** immediately (so `expert_tier.cpp` reserves space for it) and defers `DirectFile::open()` + `mmap()` until Warmup.
   - After punch-on-read reclaims `/mnt/ram/model.basalt`, `boot.sh` measures real `/proc/meminfo` `MemAvailable` and calls `basalt_jobs.py --finalize-engram` (preserving `>= 5 GiB` safety reserve): places **Layer 1 all + Layer 14 3/4-gram (`~83–85.9 GiB`)** into 2 compact `/mnt/ram/engram_*.bin` `mmap` windows, leaving only `~17–20 GiB` Layer 14 `2-gram` on sparse SSD (`+4096 B` overlap at interior boundaries so `144`/`176 B` rows straddling 4 KiB pages never read zeros). If real `MemAvailable` is tighter than estimated, only the disjoint SSD delta is fetched. **0 experts on SSD, ~80–83% Engram in RAM, 95–99% of remaining `2-gram` SSD lookups absorbed by `4 GiB` `RowCache`!**
3. **`wave+hybrid` (`UD-Q2_K_XL`, ~194.3 GiB backbone + 125.89 GiB `Q5_K` Engram — Stage 1 Micro-Wave)**:
   - Writes **0 bytes of backbone to SSD** while launching the sparse SSD Engram tail (`~83.9 GiB`) **in parallel at `t = 2s`**.
   - Pre-fetches only the non-expert base sections (`header + meta + gguf + dspark + vision`, ~8–10 GiB) into sparse `/mnt/ram/model.basalt`, applies 62/62 in-place abliteration, then runs `POST /load` with `BASALT_WAVE_FETCH_CMD=wave_fetch.sh` and `BASALT_EXPERTS_RANGE` (strictly clipped to `experts` so abliterated `dense` is never overwritten): Wave 1 streams VRAM experts → punches tmpfs; Wave 2 sorts RAM experts by ascending file offset and streams ~20 GiB micro-waves (`fetch → copy → punch`, peak RAM ≤ 141 GiB).
   - Locks **Layer 1 3/4-gram (`~42 GiB`)** into 1 RAM window via `--finalize-engram` while skipping SSD-resident Layer 14 `4-gram` from `RowCache` (`cache_skip`). See [`docs/q2k-xl-acceleration.md`](q2k-xl-acceleration.md) for the optional Stage 2 async tail design.
4. **Fail-fast guard**: Any pack whose backbone exceeds `VRAM + RAM` zero-SSD-expert capacity (e.g. `NVFP4`, ~294 GiB backbone) is rejected in 1 second before spending CU.

Boot sequence (`~95–108s` with SHA-256 verified GitHub Actions prebuilt `sm_120` tarball):
1. **Parallel**: Multi-process native C `curl | dd` (`pfetch.py`, 128 MiB RAM chunks, ~3.45+ GB/s zero-GIL) streams the sparse backbone (or base sections in `wave+hybrid`), 43 KB rank-2 `abliteration-edit.npz` (`~1s`), `vm/build.sh` (verifies `prebuilt_sha256` + patch symbols, unpacks `basalt` + `abliterate`, and installs missing Python deps pinned against `$S/requirements.txt` into `/mnt/ram/pydeps`), and (in hybrid modes) background sparse SSD download of the Engram SSD range.
2. **In-place Abliteration (`~1.5–3s`)**: `abliterate_basalt.py --scale $UNCENSOR_SCALE` + standalone OpenMP `abliterate.c` applies mrfakename's 43 KB rank-2 edit directly across all 62/62 target `Q8_0`/`Q5_K`/`Q6_K` tensors via stochastic rounding (0% decode slowdown).
3. **Load & Punch-on-Read (`~10–15s`, or micro-wave in `wave+hybrid`)**: `POST /load` moves hot experts to VRAM and resident complement to 2 MiB `MADV_HUGEPAGE` pinned RAM (upstreamed THP + parallel `cudaHostRegister`). `BASALT_PUNCH_FILE` calls `fallocate(FALLOC_FL_PUNCH_HOLE)` after VRAM and RAM waves (and per ~20 GiB chunk in `wave+hybrid`), shrinking `/mnt/ram/model.basalt` down to **~1.6 GiB**.
4. **Engram Finalize & Stream (`~24–28s`)**: In `ram+ram`, streams 103 GiB into `/mnt/ram/engram.bin`; in hybrid modes, `--finalize-engram` writes `/mnt/ram/engram_layout.txt` from real `MemAvailable`, streams compact `/mnt/ram/engram_*.bin` windows at ~3.4+ GB/s, and joins the background SSD tail.
5. **Warmup & Gate (`~1s`)**: Warmup triggers lazy multi-window `mmap()` + sparse SSD open, verifies generation, opens `gate.py` on `:18080`, and arms the VM-side hard-cap backstop (`MAX_HOURS * 3600 + 300s`).

## 2. Native Basalt v0.6.0 v2 Configuration (`basalt_colab/vm/serve.sh`)

- **Native Shared KV Pool (`v0.6.0 #808`)**: Each slot receives up to the full `basalt.max_context` (up to `1,048,576` / 1M tokens) natively from the shared KV pool without `llama.cpp`-style `parallel × context` multiplication (`slot-full-context` was removed upstream in `v0.6.0`).
- **Performance Profiles & Routing Modes**: Default `exact` (`parallel=2`, `vram-reserve=450 MiB`, 100% lossless routing aligned with upstream Basalt); `solo` (`parallel=1`, `vram-reserve=400 MiB`, 100% lossless, maximizes VRAM resident experts for single-user workloads); optional opt-in lossy `balanced` (`--route-prior 0.1 --route-tail-skip 5`).
- **Elastic FP8/FP16 KV**: Default `fp8` uses 920 B/token (`0.24 GiB` idle, `0.97 GiB` at 1M); `fp16` keeps unrounded KV (`3,200 B/token`).
- **Dynamic Vision**: `vision = "dynamic"` consumes **0 MiB idle VRAM and 0 MiB idle RAM** (identical memory plan to `vision = "off"`).
- **32 GiB Disk Conversation Cache**: `$SSD_DIR/conv_cache` snapshots evicted KV slots as `.bcs` files and restores them in `<1 ms` (99.8% hit verified across rotated multi-turn sessions).
- **7 Surgical Patches (`patch_basalt.py`, trimmed as `O_DIRECT` tmpfs fallback `#835` & `alloc_ram` THP `#834` were upstreamed in `v0.6.0`)**: `BASALT_COLAB_ENGRAM_ENV` (`BASALT_ENGRAM_FILE`, `BASALT_ENGRAM_INFLIGHT`, `BASALT_ENGRAM_CACHE_MIB`), `BASALT_COLAB_ENGRAM_SPLIT`, `BASALT_COLAB_PUNCH_ON_READ` + **`BASALT_COLAB_WAVE_FETCH`** (`expert_tier.cpp` + `loader.cpp`), `BASALT_COLAB_PUNCH_GUARD`, `BASALT_COLAB_ENGRAM_MMAP` (zero-copy `mmap()`, `BASALT_ENGRAM_LAZY_OPEN`, multi-window `BASALT_ENGRAM_LAYOUT`, and unified `RowCache::insert()` `cache_skip` filter), and `BASALT_COLAB_RAM_PARALLEL_PIN` (parallel `cudaHostRegister`).

## 3. Enhanced Local Gateway & 6-Layer Cost Protection

- **Unified Control Console (`/_console`) + Native Basalt Web UI (`/`)**: `/` proxies untouched to Basalt's built-in console once G4 is ready, and shows `/_console` while G4 is off/starting. `/_console` is authenticated with `api_key` (auto-loaded from `#key=` fragment by `basalt-colab console`), renders live Basalt telemetry, plots 14-day CU billing curves from `usage.jsonl`, and hot-reloads `config.toml` online.
- **6-Layer Cost Safety**:
  1. `0.5s` `MSG_PEEK` client-disconnect cancellation (aborts active GPU Prefill/Decode AND unwakes cold-start waiters).
  2. Orphan cold-start auto-cancel (`orphan_boot_s`, default `30s`).
  3. Idle auto-stop (`idle_min`, default `30 min`, hot-reloadable live).
  4. Proxy hard cap (`max_hours`, default `6 h`, hot-reloadable live).
  5. Detached local `_watchdog` process (survives proxy SIGKILL).
  6. VM-side hard-cap timer in `boot.sh` (kills Basalt and wipes `/content/secrets` if local machine sleeps).
