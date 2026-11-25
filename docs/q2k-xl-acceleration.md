# `UD-Q2_K_XL` 加速冷启动与上线预备方案

> **状态**：方案与实施路径已就绪，待上游条件齐备后直接按阶段落地（当前 `v0.5.7` 生产链路零侵入）。

---

## 1. 上线前置条件（Trigger Checklist）

当以下两项上游条件就绪时，即可启动本方案：

- [ ] **条件 A：Basalt 新版本发布（含 PR [`#481`](https://github.com/jesdga95/basalt/pull/481) `fcd05701` 及后续稳定性修复）**
  - 原生支持 `UD-Q2_K_XL`（`MXFP4` / `IQ3_S` down experts + `Q5_K` 176 B/row Engram + 加载期 `native_expert_supported` 校验）。
  - 同步吸收 `v0.5.7` 之后的关键服务端修复（如 `#645` 重启死锁、`#646` tool_choice grammar、`#718` SSE 信号量泄漏）。
  - 核对 [`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 在目标 tag 上的锚点，触发 [`.github/workflows/build-basalt-prebuilt.yml`](../.github/workflows/build-basalt-prebuilt.yml) 产出 `prebuilt-v0.5.8` 并回填 [`basalt_colab/vm/manifest.json`](../basalt_colab/vm/manifest.json)（含 `prebuilt_sha256`）。
- [ ] **条件 B：Hugging Face `UD-Q2_K_XL.basalt` 仓库就绪**
  - 上游 `jesdga/DeepSeek-V4.1-Flash-1017-Basalt` 发布官方包，或使用 [`scripts/build_q2k_xl_pack.py`](../scripts/build_q2k_xl_pack.py) 转换并上传至自托管 HF 仓库。
  - 回填 `manifest.json` 中 `packs.ud-q2_k_xl.url`。

---

## 2. 物理边界与性能基线（Colab G4：96 GiB VRAM + 176 GiB RAM + 220 GiB SSD）

### 2.1 实测精度与吞吐（上游 PR `#481`，单卡 96 GiB RTX PRO 6000 Blackwell，100% 无损 `exact`）

| 量化包 | 总大小 | Backbone / Engram | 96 GiB VRAM 专家数（/384） | VRAM 命中率 | KLD（vs BF16） | Top-1 | 无损 Decode 吞吐 |
|---|---:|---|---:|---:|---:|---:|---|
| `UD-IQ1_M` | 260 GB | 139.1 GiB / 103.0 GiB (`IQ4_NL`) | ~210+ | ~96–98% | 0.2779 | 83.88% | 148–160+ tok/s |
| `UD-IQ2_XXS` | 285 GB | 162.85 GiB / 103.0 GiB (`IQ4_NL`) | ~175+ | ~96–98% | 0.2408 | 85.15% | 94–125 tok/s |
| **`UD-Q2_K_XL`** | **~343 GB** (~320 GiB) | **~195–205 GiB / ~125.9 GiB (`Q5_K`)** | **143–153** | **~94–96%+** | **0.1501**（PPL 2.7356 vs BF16 2.7314） | **88.42%** | **116.5 conv / 210 struct / 100 prose tok/s** |

> 全部剩余 ~231–241 个专家（~115–120 GiB）仍可 **100% 装入 Colab G4 的 176 GiB RAM（0 专家落 SSD）**，因此推理期无损 `exact` 即可稳定 100–210 tok/s，**无需开启任何有损路由**。

### 2.2 为什么现有 `ssd+hybrid` 冷启动长达 ~18–21 分钟？

- Colab G4 实测带宽：
  - HF CDN → `/mnt/ram`（tmpfs）：**~3.45 GB/s（~3.21 GiB/s）**，可用上限 **~164 GiB**
  - HF CDN → `/content`（SSD `loop0`）：**~288 MiB/s（~0.281 GiB/s，比 tmpfs 慢 11.5×）**
- 现有 `ssd+hybrid` 因为 Backbone（~200 GiB）超出 tmpfs（164 GiB）约 36 GiB，将**整块 200 GiB Backbone 降级写往 `/content`**（~11.8 min），且只能等 `/load` punch 释盘后再串行写 ~88 GiB Engram SSD tail（~5.2 min），总冷启 **~18–21 分钟（~3.0 CU/次）**。
- **物理下限**：`VRAM (~86 GiB) + RAM (~158 GiB) = 244 GiB`，因此 326 GiB 总包中**物理上真正必须写往 `/content` 慢盘的只有 ~82–88 GiB Engram tail**，**Backbone 在物理上 0 字节需要落 SSD**！

---

## 3. 分阶段加速实施路径

### 阶段总览

| 阶段 | 方案 | 写 `/content` 总量 | 预计冷启耗时 | 改动范围 |
|---|---|---:|---:|---|
| 基线 | 当前 `ssd+hybrid` | ~288 GiB | ~18–21 min | 现有代码 |
| **Stage 0** | 探测 G4 本地 NVMe scratch 盘 | 视硬件 | 若有 NVMe：**~2.5 min** | `boot.sh`（~10 行） |
| **Stage 1（核心）** | **Loader 驱动两波按需流式加载**（0 字节 Backbone 落盘 + `t=2s` 并行下 Engram tail） | **~88 GiB**（物理下限） | **~5.5 min**（全量落盘完成） | `patch_basalt.py` + `basalt_jobs.py` + `boot.sh` |
| **Stage 2（极致）** | **Engram Tail 异步落盘 + 4 KiB HTTP Range 无损回源**（专家 + 38 GiB RAM Engram 就绪即开闸） | ~88 GiB（后台异步） | **~2.2 min 首字可用**（100% 无损） | `pfetch.py` + `patch_basalt.py` (`DirectFile`) + `boot.sh` |

---

### Stage 0：开机探测高速本地盘（零风险、先做）

- **改动位置**：[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)
- **做法**：在运行 `basalt_jobs.py --plan` 前，探测 VM 是否存在独立高速本地 NVMe（如 `/dev/nvme*n1` 未挂载分区或独立高速 mount，实测 `dd` 128 MiB 写速 > 600 MiB/s）。若存在，则将 `SSD_DIR` 从 `/content` 切至该盘（替代 288 MiB/s 的 `loop0`）；若无则无感回退 `/content`。

---

### Stage 1（核心）：Loader 驱动的两波按需流式加载（`~20 min → ~5.5 min`）

#### 设计精髓：零猜测、复用现有 `BASALT_COLAB_PUNCH_ON_READ` 两波屏障

不用在 Python 里重算 Basalt 的专家分层算法——[`patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 的 Patch `#3b`（`src/core/loader.cpp`）**原本就已经把 `POST /load` 拆成了严格的 Wave 1（VRAM `device_reads` + punch）与 Wave 2（RAM `host_reads` + punch）**，而且 `loader.cpp` 在执行前已经精确算出了每一波的合并字节区间 `collect_spans(...)`！

```mermaid
sequenceDiagram
    participant Boot as boot.sh
    participant SSD as /content SSD (288 MiB/s)
    participant RAM as /mnt/ram tmpfs (3.45 GB/s)
    participant Loader as basalt Loader (POST /load)
    participant GPU as 96 GiB VRAM + Pinned RAM

    Note over Boot,SSD: t = 2s: 启动后台流式写入 ~88 GiB Engram Tail (耗时 ~313s, 全程并行!)
    Boot->>SSD: pfetch.py engram_tail.bin (88 GiB, 后台并行)

    Note over Boot,RAM: t = 2s: 仅下载非 experts 小节 (~8-10 GiB: header+meta+gguf+dspark+vision)
    Boot->>RAM: pfetch.py sparse model.basalt (~3s, experts 保持 0-block 空洞)
    Boot->>RAM: 原地 abliterate (62 个 dense 张量全在 gguf 内, ~1.5s)
    Boot->>Loader: POST /load

    Note over Loader,GPU: Wave 1 (VRAM 波: ~80-85 GiB)
    Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 dev_spans (~25s)
    Loader->>GPU: run_wave(rounds[0]) -> 写入 96 GiB VRAM (~6s)
    Loader->>RAM: punch_spans(dev_spans) -> /mnt/ram 立刻回落至 ~2 GiB

    Note over Loader,GPU: Wave 2 (RAM 波: ~110-115 GiB)
    Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 host_spans (~35s)
    Loader->>GPU: run_wave(punch_host_wave) -> 写入 MADV_HUGEPAGE Pinned RAM (~8s)
    Loader->>RAM: punch_spans(host_spans) -> /mnt/ram 立刻回落至 ~2 GiB

    Note over Boot,RAM: t ≈ 85s: 全部 384/384 专家已 100% 在 VRAM+RAM 就绪!
    Boot->>RAM: pfetch.py ~38 GiB Engram RAM head (~12s)
    Boot->>SSD: wait PID_TAIL (Stage 1 等后台写完; Stage 2 直接开闸!)
```

#### 具体代码改动清单

1. **[`basalt_colab/vm/basalt_jobs.py`](../basalt_colab/vm/basalt_jobs.py)**
   - 新增规划模式 `wave+hybrid`（当 `bb_gib > ram_total_gib - 12.0` 且 `ram_needed_gib <= max_ram_experts_gib` 时启用，替代慢速 `ssd+hybrid`）：
     - `model_path = "/mnt/ram/model.basalt"`，`punch_file = "/mnt/ram/model.basalt"`，`parallel_tail = True`（因为 0 字节 Backbone 写 `/content`，Engram tail 可在 `t=2s` 立刻启动！）。
   - `build_jobs()` 在 `wave+hybrid` 下输出：
     - `j_basalt_bb_base.json`：仅包含 `header + meta + gguf + dspark/gguf + vision/mmproj`（排除 `experts` 与 `engram`，共 ~8–10 GiB），但 `dest_size` 仍为完整 `file_bytes`（稀疏空洞文件）。
     - `j_basalt_eng.json`（~38 GiB RAM head）与 `j_basalt_eng_tail.json`（~88 GiB SSD tail）。
2. **[`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py)**
   - 新增 Patch `#9 BASALT_COLAB_WAVE_FETCH`（位于 `src/core/loader.cpp`，紧贴现有 Patch `#3b`）：
     - 检查环境变量 `BASALT_WAVE_FETCH_CMD`（例如 `/content/vm/wave_fetch.sh`）。
     - 在 `run_wave(rounds[0])` 之前：将 `dev_spans`（排除已非空的 `gguf`/`meta` 小节或仅保留落在 `experts` 区间的 span）写入临时 JSON，执行 `BASALT_WAVE_FETCH_CMD <spans.json> wave1`，返回非 0 则 fail-fast。
     - `run_wave(rounds[0])` 完成后立刻 `punch_spans(dev_spans)`，释放 ~85 GiB tmpfs。
     - 在 `run_wave(punch_host_wave)` 之前：将 `host_spans` 写入临时 JSON，执行 `BASALT_WAVE_FETCH_CMD <spans.json> wave2`；完成后 `punch_spans(host_spans)`。
   - **顺带验证 Patch `#3a`（`expert_tier.cpp`）**：由于 `/mnt/ram/model.basalt` 此时仅实际占用 ~8–10 GiB，主机 `MemAvailable` 本身就接近 ~160 GiB，无需过度依赖 `punch_credit`，规划更准。
3. **[`basalt_colab/vm/wave_fetch.sh`](../basalt_colab/vm/)（新增 ~20 行脚本）**
   - 读取 `spans.json`，生成 `pfetch.py` job 列表（按 `src_off = dest_off = span.start, size = span.end - span.start` 直接从 `MODEL_URL` 拉取到 `/mnt/ram/model.basalt`），以 `--workers $WORKERS --chunk-mb 128` 满速写入。
4. **[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)**
   - `wave+hybrid` 下自动设置 `BASALT_WAVE_FETCH_CMD=/content/vm/wave_fetch.sh`、在 `t=2s` 启动 `PID_TAIL` 并行下 Engram SSD tail、前置仅拉 `j_basalt_bb_base.json`（~3s）→ 原地 abliteration（~1.5s）→ `POST /load`（内部自动跑 Wave 1 + Wave 2）→ 拉 ~38 GiB RAM Engram head（~12s）→ `wait $PID_TAIL`。

#### 收益与风险评估
- **全量就绪时间**：`~20 s` (Colab 分配) + `max(95 s [Wave1+Wave2+RAM Engram], 313 s [88 GiB SSD tail])` + `5 s` (warmup) ≈ **`~5.5 分钟`**（比 20 分钟提速 **3.6×**，冷启 CU 从 ~3.0 降至 **~0.8 CU**）。
- **额外红利**：该机制对 `UD-IQ1_M` / `UD-IQ2_XXS` 同样适用，tmpfs 峰值占用从 `163 GiB` 降至 `~115 GiB`，且彻底消灭“Backbone 略超 RAM 就坠崖”的问题。
- **稳定性**：100% 本地闭环，开闸后零外部网络依赖，风险极低。

---

### Stage 2（极致体验）：Engram Tail 异步落盘 + 无损 HTTP Range 回源（`~5.5 min → ~2.2 min` 首字可用）

在 Stage 1 基础上，`t ≈ 115 s` 时**全部 384/384 专家 + ~38 GiB 热区 Engram 均已在 VRAM/RAM 就绪**，唯一在等的是后台还要写 ~3 分钟的 ~88 GiB 冷区 Engram tail。

#### 实现路径

1. **[`basalt_colab/vm/pfetch.py`](../basalt_colab/vm/pfetch.py)**
   - 增加可选参数 `--ready-bitmap /mnt/ram/engram_tail.bits`（按 64 MiB chunk 分配，88 GiB 仅需 `1,408 bits ≈ 176 bytes` + 8-byte header）。
   - 每写完并 `fsync`/落地一个 64 MiB chunk，原子置位对应 bit；全部完成后置 `all_ready = 1`。
2. **[`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py)（扩展 Patch `#2b DirectFile`）**
   - 当设置了 `BASALT_ENGRAM_TAIL_BITMAP` 和 `BASALT_ENGRAM_TAIL_URL`（以及 `BASALT_ENGRAM_TAIL_SRC_OFF`）：
     - `DirectFile` 读 `fd2`（tail）时先做一次无锁原子 bit 检测（开销 < 5 ns）：
       - **已落盘（`bit == 1` 或 `all_ready == 1`）**：走原始 `pread(fd2, ...)`，零开销；
       - **尚未落盘（仅开机前 3 分钟内的冷区未命中行）**：通过轻量本地 HTTP Range 代理或 `libcurl` 长连接同步拉取精确 `[src_off + tail_off, len)`（单行 176 B / 4 KiB，耗时 ~5–15 ms，内容与最终落盘字节**完全逐比特一致**），结果照常进入 `RowCache`（1024 MiB）避免重复回源。
3. **[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)**
   - `t ≈ 115 s`（Wave 1 + Wave 2 + RAM Engram head 完成）直接 `touch $R/READY` → Warmup → 开闸 `gate.py`，不再阻塞 `wait $PID_TAIL`；`PID_TAIL` 在后台静默写完剩余 tail 后退出，日志记录 `engram tail background complete`。

#### 收益与权衡
- **首字可用时间**：**~2.0–2.3 分钟**（与 `UD-IQ2_XXS` 基本一致），**100% 无损（逐比特精确）**。
- **权衡**：开机后前 3 分钟内若触发尚未落盘的冷区 N-gram，个别 token 会多 ~5–15 ms 回源延迟；3 分钟后后台落盘完成自动归零。需为 `basalt` 链接 `libcurl` 或通过本机常驻 Python HTTP/1.1 Keep-Alive Unix socket 代理（零 C++ 外部依赖，更符合本项目品味）。

---

## 4. 升级与验证步骤（待上游就绪后按序执行）

1. **升级 Basalt 基线**：
   - 将 [`manifest.json`](../basalt_colab/vm/manifest.json) 指向新 tag（`v0.5.8`）及对应 commit；
   - 在本地或 CI 验证 `patch_basalt.py` 对新 tag 全部 8 个现有 patch 锚点 100% 匹配；
   - 触发 `.github/workflows/build-basalt-prebuilt.yml` 发布 `prebuilt-v0.5.8` 并锁定 `prebuilt_sha256`。
2. **接入 `UD-Q2_K_XL` 包直链**：
   - 回填 `manifest.json` `packs.ud-q2_k_xl.url`；
   - 放行 `config.py` 对 `ud-q2_k_xl` 免填 `--pack-url`（沿用 `manifest.json` 内置 URL）。
3. **先落地 Stage 0 + Stage 1，单次真机 G4 实测验收**：
   - 核验 `timeline.txt`：Wave 1 / Wave 2 / punch 耗时、`/mnt/ram` 峰值、`/content` 写入总量（确认 0 字节 Backbone 落盘）；
   - 核验 `basalt` 启动日志：`expert homes`（目标 143–153/384 in VRAM，0 on SSD）、`0 lossy flags`、实测 `conv / struct / prose` 吞吐与 `hit %`。
4. **按需叠加 Stage 2**：
   - 若希望将 ~5.5 min 进一步压至 ~2.2 min，再实现 `ready_bitmap` + Unix-socket Keep-Alive Range 回源，并用 `test_config.py` 本地单测验证未就绪 chunk 回源与就绪 chunk 切换的逐字节一致性。
