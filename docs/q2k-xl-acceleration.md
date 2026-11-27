# `UD-Q2_K_XL` 加速冷启动与推理期 Engram 优化蓝图

> **状态**：方案已完成架构复审（含 Wave 2 峰值内存校验与推理期 Engram IOPS 分析），待上游条件齐备后直接按阶段落地（当前 `v0.5.7` 生产链路零侵入）。

---

## 1. 上线前置条件（Trigger Checklist）

当以下两项上游条件就绪时，即可启动本方案：

- [ ] **条件 A：Basalt 新版本发布（含 PR [`#481`](https://github.com/jesdga95/basalt/pull/481) `fcd05701` 及后续稳定性修复）**
  - 原生支持 `UD-Q2_K_XL`（`MXFP4` / `IQ3_S` down experts + `Q5_K` 176 B/row Engram + 加载期 `native_expert_supported` 校验）。
  - 同步吸收 `v0.5.7` 之后的关键服务端修复（如 `#645` 重启死锁、`#646` tool_choice grammar、`#718` SSE 信号量泄漏）。
  - 核对 [`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 在目标 tag 上的锚点，触发 [`.github/workflows/build-basalt-prebuilt.yml`](../.github/workflows/build-basalt-prebuilt.yml) 产出 `prebuilt-v0.5.8` 并回填 [`basalt_colab/vm/manifest.json`](../basalt_colab/vm/manifest.json)（含 `prebuilt_sha256`）。
- [ ] **条件 B：Hugging Face `UD-Q2_K_XL.basalt` 仓库就绪**
  - 上游 `jesdga/DeepSeek-V4.1-Flash-1017-Basalt` 发布官方包，或使用 [`scripts/build_q2k_xl_pack.py`](../scripts/build_q2k_xl_pack.py)（在 `v0.5.8` 发版前需加 `--basalt-ref main`）转换并上传至自托管 HF 仓库。
  - 回填 `manifest.json` 中 `packs.ud-q2_k_xl.url`。

---

## 2. 物理边界与性能基线（Colab G4：96 GiB VRAM + 176 GiB RAM + 220 GiB SSD）

### 2.1 实测精度、精确体积与吞吐（上游 PR `#481`，单卡 96 GiB RTX PRO 6000 Blackwell，100% 无损 `exact`）

实查 `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`（7 个 `UD-Q2_K_XL` 分片 + `dspark-Q8_0` + `mmproj-BF16`）精确体积为 **`320.14 GiB`（343.7 GB）**，其中 `Q5_K` Engram 精确占 **`125.89 GiB`**，Backbone 实际仅 **`~194.25 GiB`**。

| 量化包 | 总大小 | Backbone / Engram | 96 GiB VRAM 专家数（/384） | VRAM 命中率 | KLD（vs BF16） | Top-1 | 无损 Decode 吞吐 |
|---|---:|---|---:|---:|---:|---:|---|
| `UD-IQ1_M` | 260 GB | 139.1 GiB / 103.0 GiB (`IQ4_NL`) | ~210+ | ~96–98% | 0.2779 | 83.88% | 148–160+ tok/s |
| `UD-IQ2_XXS` | 285 GB | 162.85 GiB / 103.0 GiB (`IQ4_NL`) | ~175+ | ~96–98% | 0.2408 | 85.15% | 94–125 tok/s |
| **`UD-Q2_K_XL`** | **343.7 GB** (`320.14 GiB`) | **`~194.3 GiB` / `125.89 GiB` (`Q5_K`)** | **143–153** | **~94–96%+** | **0.1501**（PPL 2.7356 vs BF16 2.7314） | **88.42%** | **116.5 conv / 210 struct / 100 prose tok/s** |

> 全部剩余 ~231–241 个专家（`~106–112 GiB`）可 **100% 装入 Colab G4 的 176 GiB RAM（0 专家落 SSD）**，因此推理期无损 `exact` 即可稳定 100–210 tok/s，**无需开启任何有损路由**。

### 2.2 为什么现有 `ssd+hybrid` 冷启动长达 ~18–21 分钟？

- Colab G4 实测带宽：
  - HF CDN → `/mnt/ram`（tmpfs）：**~3.45 GB/s（~3.21 GiB/s）**，可用上限 **~164 GiB**
  - HF CDN → `/content`（SSD `loop0`）：**~288 MiB/s（~0.281 GiB/s，比 tmpfs 慢 11.5×）**
- 现有 `ssd+hybrid` 因为 Backbone（`~194.3 GiB`）超出 tmpfs（`164 GiB`）约 30 GiB，将**整块 194 GiB Backbone 降级写往 `/content`**（~11.5 min），且只能等 `/load` punch 释盘后再串行写 Engram SSD tail（~4.5–5 min），总冷启 **~18–21 分钟（~3.0 CU/次）**。
- **物理下限**：`VRAM (~86 GiB) + RAM (~160 GiB) = 246 GiB`，因此 `320.14 GiB` 总包中**物理上真正必须写往 `/content` 慢盘的只有 `~66–76 GiB` Engram（取决于 `RowCache` 配额）**，**Backbone 在物理上 0 字节需要落 SSD**！

---

## 3. 分阶段冷启动加速实施路径

### 阶段总览

| 阶段 | 方案 | 写 `/content` 总量 | 峰值 RAM 占用 | 预计冷启耗时 | 改动范围 |
|---|---|---:|---:|---:|---|
| 基线 | 当前 `ssd+hybrid` | ~270–282 GiB | ~160 GiB | ~18–21 min | 现有代码 |
| **Stage 0** | 探测 G4 本地 NVMe scratch 盘 | 视硬件 | ~160 GiB | 若有 NVMe：**~2.2 min** | `boot.sh`（~10 行） |
| **Stage 1（核心）** | **Loader 驱动微波分块流式加载（Micro-Wave）**（0 字节 Backbone 落盘 + `t=2s` 并行下稀疏 Engram SSD 段） | **`~68–76 GiB`**（物理下限） | **`~134 GiB`**（防 OOM） | **`~4.3–4.8 min`**（全量就绪） | `patch_basalt.py` + `basalt_jobs.py` + `wave_fetch.sh` + `boot.sh` |
| **Stage 2（极致）** | **Engram SSD 段读优先异步落盘 + 4 KiB HTTP Range 无损回源**（专家 + RAM Window 就绪即开闸） | ~68–76 GiB（后台 Idle 写） | ~134 GiB | **~2.0–2.3 min 首字可用**（100% 无损） | `pfetch.py` + `patch_basalt.py` (`DirectFile`) + `boot.sh` |

---

### Stage 0：开机探测高速本地盘（零风险、先做）

- **改动位置**：[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)
- **做法**：在运行 `basalt_jobs.py --plan` 前，探测 VM 是否存在独立高速本地 NVMe（如 `/dev/nvme*n1` 未挂载分区或独立高速 mount，实测 `dd` 128 MiB 写速 > 600 MiB/s）。若存在，则将 `SSD_DIR` 从 `/content` 切至该盘（替代 288 MiB/s 的 `loop0`）；若无则无感回退 `/content`。

---

### Stage 1（核心）：Loader 驱动的微波分块流式加载（`~20 min → ~4.3–4.8 min`）

#### 关键架构修正：为什么 Wave 2 必须“微波分块（~16–20 GiB/批）”而不是整波拉取？

- 如果 Wave 2 一次性将全部 `~108 GiB` `host_spans` 拉入 `/mnt/ram/model.basalt`，再一次性拷入 `alloc_ram()` 后才 `punch_spans`，瞬时物理内存峰值会达到 **`108 GiB (tmpfs) + 108 GiB (alloc_ram) + 8 GiB (OS/base) ≈ 224 GiB > 176 GiB`，必然触发 OOM Kill**！
- **解法（Micro-Wave Streaming）**：在 [`patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) Patch `#3b`（`src/core/loader.cpp`）中，将 `punch_host_wave` 按每批 **~16–20 GiB** 拆成 5–6 个微波循环：`BASALT_WAVE_FETCH_CMD` 拉 20 GiB → `run_wave(sub_wave)` 拷入 `alloc_ram` → 立刻 `punch_spans(sub_spans)` 释放 20 GiB tmpfs！
- **效果**：加载期物理内存峰值从 `224 GiB` 骤降至 **`~134 GiB`**（保留 > 40 GiB 安全余量给 `RowCache` 与系统缓冲），且对 `UD-IQ1_M` / `UD-IQ2_XXS` 同样适用（tmpfs 峰值永远 ≤ ~20–40 GiB）；后续还可轻易扩展为双缓冲（拉取第 `i+1` 微波与拷贝第 `i` 微波重叠）。

```mermaid
sequenceDiagram
    participant Boot as boot.sh
    participant SSD as /content SSD (288 MiB/s)
    participant RAM as /mnt/ram tmpfs (3.45 GB/s)
    participant Loader as basalt Loader (POST /load)
    participant GPU as 96 GiB VRAM + Pinned RAM

    Note over Boot,SSD: t = 2s: 启动后台流式写入 ~72 GiB 稀疏 Engram SSD 段 (~255s, 全程并行!)
    Boot->>SSD: pfetch.py 稀疏 engram_tail.bin (Layer1 2-gram + Layer14, 后台并行)

    Note over Boot,RAM: t = 2s: 仅下载非 experts 小节 (~8-10 GiB: header+meta+gguf+dspark+vision)
    Boot->>RAM: pfetch.py sparse model.basalt (~3s, experts 保持 0-block 空洞)
    Boot->>RAM: 原地 abliterate (62 个 dense 张量全在 gguf 内, ~1.5s)
    Boot->>Loader: POST /load

    Note over Loader,GPU: Wave 1 (VRAM 波: ~85-88 GiB, 可按 32 GiB 微波)
    Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 dev_spans (~26s)
    Loader->>GPU: run_wave(rounds[0]) -> 写入 96 GiB VRAM (~6s)
    Loader->>RAM: punch_spans(dev_spans) -> /mnt/ram 立刻回落至 ~2 GiB

    Note over Loader,GPU: Wave 2 (RAM 微波循环: 5-6 批 × ~20 GiB, 峰值 RAM <= 134 GiB!)
    loop 每批 ~20 GiB host_spans
        Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 sub_spans (~6s)
        Loader->>GPU: run_wave(sub_wave) -> 写入 MADV_HUGEPAGE Pinned RAM (~1.5s)
        Loader->>RAM: punch_spans(sub_spans) -> 立刻释回 20 GiB tmpfs
    end

    Note over Boot,RAM: t ≈ 90s: 全部 384/384 专家已 100% 在 VRAM+RAM 就绪!
    Boot->>RAM: pfetch.py ~50-54 GiB Engram RAM Window (~16s)
    Boot->>SSD: wait PID_TAIL (Stage 1 等后台写完; Stage 2 直接开闸!)
```

#### 具体代码改动清单

1. **[`basalt_colab/vm/basalt_jobs.py`](../basalt_colab/vm/basalt_jobs.py)**
   - 消除双重 RAM 预留（原 `auto_budget` `+8 GiB` 与 `_split_engram_bytes` `reserve_gib=14 GiB` 叠加浪费 ~22 GiB RAM），统一为单一安全预留（~6–8 GiB）并显式扣除 `ENGRAM_CACHE_MIB`（`RowCache` 预算），避免挤压专家或 OOM。
   - 新增规划模式 `wave+hybrid`（当 `bb_gib > ram_total_gib - 12.0` 且 `ram_needed_gib <= max_ram_experts_gib` 时启用，替代慢速 `ssd+hybrid`）：
     - `model_path = "/mnt/ram/model.basalt"`，`punch_file = "/mnt/ram/model.basalt"`，`parallel_tail = True`（0 字节 Backbone 写 `/content`，Engram SSD 段在 `t=2s` 立刻启动！）。
     - `j_basalt_bb_base.json`：仅包含 `header + meta + gguf + dspark/gguf + vision/mmproj`（排除 `experts` 与 `engram`，共 ~8–10 GiB），`dest_size` 仍为完整 `file_bytes`（稀疏空洞文件）。
     - 输出 **RAM Window `[ram_lo, ram_hi)`**（`j_basalt_eng.json`）与 **逻辑全长稀疏 SSD 段**（`j_basalt_eng_tail.json`，参见第 4.3 节）。
2. **[`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py)**
   - 新增 Patch `#9 BASALT_COLAB_WAVE_FETCH`（位于 `src/core/loader.cpp`，紧贴现有 Patch `#3b`）：
     - 检查环境变量 `BASALT_WAVE_FETCH_CMD`（例如 `/content/vm/wave_fetch.sh`）与 `BASALT_WAVE_CHUNK_GIB`（默认 `20`）。
     - Wave 1：对 `rounds[0]` 按块（或整波）导出落在 `experts` 区间的 `dev_spans` 到临时 JSON → 执行 `BASALT_WAVE_FETCH_CMD <spans.json> wave1` → `run_wave` → `punch_spans`。
     - Wave 2：将 `punch_host_wave` 按 `BASALT_WAVE_CHUNK_GIB`（~20 GiB）切分为若干 `sub_wave`，每批严格执行 `fetch(sub_spans)` → `run_wave(sub_wave)` → `punch_spans(sub_spans)`，确保物理内存峰值不超过 `host_bytes + 20 GiB + 8 GiB`。
   - **顺带验证 Patch `#3a`（`expert_tier.cpp`）**：由于 `/mnt/ram/model.basalt` 此时仅实际占用 ~8–10 GiB，主机 `MemAvailable` 本身就接近 ~160 GiB，规划更准。
3. **[`basalt_colab/vm/wave_fetch.sh`](../basalt_colab/vm/)（新增 ~20 行脚本）**
   - 读取 `spans.json`，生成 `pfetch.py` job 列表（按 `src_off = dest_off = span.start, size = span.end - span.start` 直接从 `MODEL_URL` 拉取到 `/mnt/ram/model.basalt`），以 `--workers $WORKERS --chunk-mb 128` 满速写入。
4. **[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)**
   - `wave+hybrid` 下自动设置 `BASALT_WAVE_FETCH_CMD=/content/vm/wave_fetch.sh`、在 `t=2s` 启动 `PID_TAIL` 并行下稀疏 Engram SSD 段、前置仅拉 `j_basalt_bb_base.json`（~3s）→ 原地 abliteration（~1.5s）→ `POST /load`（内部自动跑 Wave 1 + Wave 2 微波）→ 拉 Engram RAM Window（~16s）→ `wait $PID_TAIL`。

#### 收益与风险评估
- **全量就绪时间**：`~20 s` (Colab 分配) + `max(105 s [Wave1+Wave2微波+RAM Window], 255 s [~72 GiB SSD 段])` + `5 s` (warmup) ≈ **`~4.3–4.8 分钟`**（比 20 分钟提速 **4.3×**，冷启 CU 从 ~3.0 降至 **~0.68 CU**）。
- **稳定性**：100% 本地闭环，峰值 RAM ≤ 134 GiB 绝不 OOM，开闸后零外部网络依赖，风险极低。

---

### Stage 2（极致体验）：读优先异步落盘 + 无损 HTTP Range 回源（`~4.6 min → ~2.2 min` 首字可用）

在 Stage 1 基础上，`t ≈ 115 s` 时**全部 384/384 专家 + ~50 GiB RAM Window Engram 均已在 VRAM/RAM 就绪**，唯一在等的是后台还要写 ~2.5 分钟的 ~72 GiB SSD Engram 段。

#### 实现路径（含读写竞争防饿死保护）

1. **[`basalt_colab/vm/pfetch.py`](../basalt_colab/vm/pfetch.py)**
   - 增加可选参数 `--ready-bitmap /mnt/ram/engram_tail.bits`（按 64 MiB chunk 分配，126 GiB 逻辑空间仅需 `2,016 bits ≈ 252 bytes` + 8-byte header）。
   - 每写完并 `fsync` 一个 64 MiB chunk，原子置位对应 bit；全部完成后置 `all_ready = 1`。
   - **下载优先级**：优先落盘 **Layer 1 2-gram**（若在 SSD），再落盘 Layer 14 2-gram / 3-gram / 4-gram。
   - **读优先退让（关键防坑！）**：开闸前（`! -f $R/READY`）满速写；检测到 `$R/READY` 存在后自动降级为 `ionice -c 3`（Idle IO 调度）并限并发/限速（如 `--workers 4` / ~120–150 MiB/s），**严防后台满载写把 GCE PD 的在线 4 KiB 随机读 p99 延迟打爆 5–10×**！
2. **[`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py)（扩展 Patch `#2b DirectFile`）**
   - 当设置了 `BASALT_ENGRAM_TAIL_BITMAP` 和 `BASALT_ENGRAM_TAIL_URL`（以及 `BASALT_ENGRAM_TAIL_SRC_OFF`）：
     - `DirectFile` 读 `fd2` 时先做一次无锁原子 bit 检测（开销 < 5 ns）：
       - **已落盘（`bit == 1` 或 `all_ready == 1`）**：走原始 `pread(fd2, ...)`，零开销；
       - **尚未落盘（仅开机前 ~2.5 分钟内的未命中行）**：通过本机常驻 Python HTTP/1.1 Keep-Alive Unix socket 代理（零 C++ 外部依赖）同步拉取精确 `[src_off + off, len)`（耗时 ~5–15 ms，内容逐比特无损），结果照常进入 `RowCache` 避免重复回源。
3. **[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)**
   - `t ≈ 115 s`（Wave 1 + Wave 2 + RAM Window 完成）直接 `touch $R/READY` → Warmup → 开闸 `gate.py`，不再阻塞 `wait $PID_TAIL`；`PID_TAIL` 在后台以 Idle I/O 静默写完后退出。

---

## 4. 推理期 Engram `/content` 随机读 IOPS 瓶颈分析与优雅优化

除冷启动写盘带宽外，Colab `/content` 虚拟盘（GCE PD / `loop0`，典型 4 KiB 随机读约 **3,000–15,000 IOPS**、单次延迟 **0.5–2 ms**、p99 **5–15 ms**）在 `UD-Q2_K_XL` **推理期** 还会引入显著的 Engram 随机读 IOPS 风险。（注：本地中高端 PCIe 4.0/5.0 NVMe 具备 500K–1M IOPS、0.05–0.2 ms 延迟，几乎不受影响；本节针对的是 Colab 虚拟盘及低 IOPS 云盘/SATA 盘。）

### 4.1 真实 Engram 表布局与各量化包对比（实查 `BASALTMF v2` 头）

- **结构**：共 2 层 Engram（**Layer 1** 与 **Layer 14**），合计 `768,022,850` 行；每 token 查 `n_gram=4`（2/3/4-gram 共 3 阶）× `8 heads` = **每层 24 次、合计 48 次均匀哈希随机读 / token**。
- **读放大**：每 head 约 `1,600 万` 行，素数取模无空间局部性；`DirectFile` 对 `/content` 走 `O_DIRECT` 4 KiB 随机 `pread` 取 `176 B` 行（**23.3× 读放大**）。
- **精确区间布局（`Q5_K`，176 B/行）**：
  - `[0.00 .. 20.98 GiB)`：Layer 1 `2-gram`（最热，`RowCache` 命中率 60–85%）
  - **`[20.98 .. 62.94 GiB)`：Layer 1 `3-gram` + `4-gram`（共 `41.96 GiB`，恰好一整块连续区间！）**
  - `[62.94 .. 83.92 GiB)`：Layer 14 `2-gram`（热）
  - `[83.92 .. 125.89 GiB)`：Layer 14 `3-gram` + `4-gram`（冷，有 14 层计算遮盖）

| 量化包 | Engram 格式 | Layer 1 | Layer 14 | 总计 | G4 旧线性切分下 `/content` 落盘情况 |
| --- | --- | ---: | ---: | ---: | --- |
| `UD-IQ1_M` | `IQ4_NL` (144 B/行) | 51.50 GiB | 51.50 GiB | 103.00 GiB | **0 GiB 落盘**（100% RAM `mmap`） |
| `UD-IQ2_XXS` | `IQ4_NL` (144 B/行) | 51.50 GiB | 51.50 GiB | 103.00 GiB | **Layer 1 100% 在 RAM**；仅 Layer 14 后 ~25 GiB 落盘，由 14 层计算遮盖 |
| **`UD-Q2_K_XL`** | `Q5_K` (176 B/行) | **62.94 GiB** | **62.95 GiB** | **125.89 GiB** | **旧切分下 Layer 1 掉盘 ~25 GiB（恰是最冷 4-gram）+ Layer 14 100% 掉盘** |

### 4.2 `/content` IOPS 瓶颈的三大体现

1. **Decode 期“Layer 1 零遮盖”同步卡顿（最致命）**
   - Layer 14 前有 14 层（Layer 0..13，~2–4 ms）GPU+RAM 计算遮盖异步 I/O，而 **Layer 1 前仅有 Layer 0（~0.15–0.25 ms）**，GPU 立刻撞上 `collect()` 同步屏障。
   - 若 Layer 1 的冷 4-gram 掉盘，DSpark 窗口 `W=4..6` 每次触发 ~15–40 次并发 4 KiB 盘读，屏障必须等该批最慢的一次（`max(15..40)` p99 约 **3–12 ms**），直接吃掉 **20%–50%+** Decode 吞吐并造成明显抖动。
2. **旧线性切分的“反向优化”效应**
   - 从 `offset 0` 线性切分时，RAM Head 先装了最容易被 `RowCache` 命中的 Layer 1 `2-gram`（21 GiB），反而把几乎 100% 打穿缓存的 **Layer 1 `4-gram`（21 GiB）** 扔到了 `/content` 慢盘。
3. **长 Prompt Prefill 与 `parallel=2` 并发撞墙**
   - `16,384` token Prefill chunk 需查 `786,432` 行；扣除去重与 `RowCache` 后仍需 **20 万–30 万次 4 KiB 随机读**，在 5K–10K IOPS 虚拟盘上仅查 Engram 就耗时 **20–60s**（Prefill 从 2,500+ tok/s 跌至 **250–800 tok/s**）；且 `model.cpp` 当前硬编码 `PleReader` `max_inflight = 64` 限制了并发排队深度。

### 4.3 优雅统一的推理期优化方案（100% 逐比特无损）

1. **单区间 RAM Window `[ram_lo, ram_hi)` + 逻辑全长稀疏 SSD 文件（核心重构，比前缀切分更简洁）**
   - `/content/engram_tail.bin` 创建为逻辑全长（`125.89 GiB`）的**稀疏文件**，只下载不在 RAM Window 里的区间（RAM 区间保持 0-block 空洞，0 字节重复落盘）；
   - `/mnt/ram/engram.bin` 紧致存放 `[ram_lo, ram_hi)`（通过环境变量 `BASALT_ENGRAM_RAM_OFF` 指定起点，默认 `0` 保持 100% 向下兼容）：
     - `PleReader` 命中 `[ram_lo, ram_hi)` → `mmap_head[at - ram_lo]`（80 ns 零拷贝）；
     - 未命中 → `DirectFile` 直接 `pread(fd2, buf, len, at)`（**无需再做 `offset - split_bytes` 偏移换算**！）。
   - **放置策略**：
     - 若可用 RAM ≥ `63 GiB`（如 `solo` 模式 + 去除双重预留）：设 `[0, 62.94 GiB)`，**Layer 1 100% 在 RAM（0 次 Layer 1 盘读）**；
     - 若可用 RAM 为 `46–56 GiB`（如 `parallel=2` 且留出 `4–8 GiB RowCache`）：设 `ram_lo = 20.98 GiB`（Layer 1 `3-gram` 起点），将 **Layer 1 `3-gram + 4-gram`（`41.96 GiB`）** 全部锁进 RAM，剩余容量顺延覆盖 Layer 14 或回吞 Layer 1 `2-gram`；仅将最易被 `RowCache` 命中的 `2-gram` 留在 SSD，**可将 Layer 1 有效盘读 IOPS 砍掉 70%–85%+**。
2. **扩大 `RowCache` 至 `4–8 GiB` + 5 行“防 4-gram 冲刷”准入过滤（拒绝 `PageCache`）**
   - **保持 `/content` 走 `O_DIRECT`（0 `PageCache`）**：`PageCache` 以 4096 B 页缓存 176 B 均匀哈希行，空间浪费高达 **95.7%（密度比 `RowCache` 低 23×）**，且易与 `Pinned RAM`/`tmpfs` 争抢内存触发 Colab cgroup direct reclaim 或 OOM Kill。
   - **放宽 Patch `#7` `RowCache` clamp**：将 [`patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 的 `4096 MiB` 上限放宽至 `16384 MiB`，`UD-Q2_K_XL` 默认配 **`4096–8192 MiB`**（存 `2,400–4,800 万` 行，等效 **`92–184 GiB PageCache`**，并从 RAM Window 预算中显式扣除）。
   - **5 行防冲刷准入过滤**：因 `RowCache` 为组内 round-robin 替换，在 `PleReader` 插入前跳过落在 SSD 的 **Layer 14 `4-gram`** 行号区间，防止长 Prompt Prefill 一次性冷 4-gram 冲刷掉高频 2/3-gram 热缓存。
3. **Colab G4 176 GiB 黄金配比（`UD-Q2_K_XL`）**
   - **Pinned RAM 专家**：`~106–112 GiB`（第一优先级，0 专家落盘）
   - **OS / KV / Prefill / Vision 安全缓冲**：`~6–8 GiB`
   - **`RowCache`**：`4–8 GiB`（横跨 Layer 1 & 14 拦截高频 2/3-gram）
   - **静态 `/mnt/ram` RAM Window (`mmap`)**：`~48–56 GiB`（全覆盖 Layer 1 `3-gram + 4-gram` `41.96 GiB`，`solo` 下逼近全覆盖 Layer 1 `62.94 GiB`）
   - **`PageCache`**：`0`（保持 `O_DIRECT`）
4. **放开 `PleReader` `max_inflight`（64 → 256）+ Stage 0 探测本地 NVMe**
   - 在 [`patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 中将 `engram_->open(..., 64, ...)` 改为可由 `BASALT_ENGRAM_INFLIGHT` 配置（默认 `256`，匹配 `DirectFile` 256 线程），提升 16K Prefill 并发队列深度；若 Stage 0 命中本地 NVMe 则 IOPS 瓶颈直接归零。
5. **备选（自转混合包，几乎无损）**：`UD-Q2_K_XL` 专家 + 4-bit Engram（`MXFP4` 136 B/行共 `97.3 GiB`，**Layer 1 仅 `48.6 GiB`**；或 `IQ4_NL` 共 `103 GiB`），立省 **23–29 GiB RAM**，`parallel=2` 下也能 100% 装下整个 Layer 1 + 部分 Layer 14。

---

## 5. 升级与验证步骤（待上游就绪后按序执行）

1. **升级 Basalt 基线**：
   - 将 [`manifest.json`](../basalt_colab/vm/manifest.json) 指向新 tag（`v0.5.8`）及对应 commit；
   - 在本地或 CI 验证 `patch_basalt.py` 对新 tag 全部现有 patch 锚点 100% 匹配；
   - 触发 `.github/workflows/build-basalt-prebuilt.yml` 发布 `prebuilt-v0.5.8` 并锁定 `prebuilt_sha256`。
2. **接入 `UD-Q2_K_XL` 包直链**：
   - 回填 `manifest.json` `packs.ud-q2_k_xl.url`；
   - 放行 `config.py` 对 `ud-q2_k_xl` 免填 `--pack-url`（沿用 `manifest.json` 内置 URL）。
3. **先落地 Stage 0 + Stage 1（Micro-Wave）+ Engram 推理优化（RAM Window + `4–8 GiB` 过滤 `RowCache` + `max_inflight=256`），单次真机 G4 实测验收**：
   - 核验 `timeline.txt`：Wave 1 / Wave 2 各微波耗时与 punch 回收、`/mnt/ram` 与整体内存峰值（目标 ≤ 135 GiB、绝不 OOM）、`/content` 写入总量（确认 0 字节 Backbone 落盘、仅 ~68–76 GiB 稀疏 Engram）；
   - 核验 `basalt` 启动日志：`expert homes`（目标 143–153/384 in VRAM，0 on SSD）、`0 lossy flags`、RAM Window 区间（确认覆盖 Layer 1 `3-gram+4-gram` 或全 Layer 1）；
   - 核验推理吞吐与 Engram I/O：实测 `conv / struct / prose` Decode 吞吐、16K 长 Prompt Prefill 吞吐以及 `PleReader` 读盘延迟（p50/p99）与 `RowCache` 命中率。
4. **按需叠加 Stage 2（读优先异步落盘）**：
   - 若希望将 ~4.5 min 进一步压至 ~2.2 min，再实现 `ready_bitmap` + Unix-socket Keep-Alive Range 回源 + `$R/READY` 后的 `ionice -c 3` 限速退让，并用 `test_config.py` 本地单测验证未就绪 chunk 回源、就绪切换与稀疏文件偏移的逐字节一致性。
