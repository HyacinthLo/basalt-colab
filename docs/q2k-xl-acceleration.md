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
- **物理下限**：`VRAM (~86 GiB) + RAM (176 GiB 扣除 ~9 GiB OS/残余 + 4 GiB RowCache + ≥5 GiB 动态安全垫 = ~158 GiB)`，因此 `320.14 GiB` 总包中**物理上必须写往 `/content` 慢盘的只有 `~78–84 GiB` Engram（标准推荐锁定 `41.96 GiB` Layer 1 3/4-gram 进 RAM 时为 `83.93 GiB`）**，**Backbone 在物理上 0 字节需要落 SSD**！

---

## 3. 分阶段冷启动加速实施路径

### 阶段总览

| 阶段 | 方案 | 写 `/content` 总量 | 峰值 RAM 占用 | 预计冷启耗时 | 改动范围 |
|---|---|---:|---:|---:|---|
| 基线 | 当前 `ssd+hybrid` | ~278–282 GiB | ~160 GiB | ~18–21 min | 现有代码 |
| **Stage 0** | 探测 G4 本地 NVMe scratch 盘 | 视硬件 | ~160 GiB | 若有 NVMe：**~2.2 min** | `boot.sh`（~10 行） |
| **Stage 1（核心）** | **Loader 驱动微波分块流式加载（Micro-Wave）**（0 字节 Backbone 落盘 + `t=2s` 并行下稀疏 Engram SSD 段） | **`~78–84 GiB`**（安全物理下限） | **`~141 GiB`**（含 4 GiB RowCache，防 OOM） | **`~5.0–5.3 min`**（全量就绪） | `patch_basalt.py` + `basalt_jobs.py` + `wave_fetch.sh` + `boot.sh` |
| **Stage 2（极致）** | **Engram SSD 段读优先异步落盘 + 4 KiB HTTP Range 无损回源**（专家 + RAM Window 就绪即开闸） | ~78–84 GiB（后台 Idle 写） | ~141 GiB | **~2.0–2.3 min 首字可用**（100% 无损） | `pfetch.py` + `patch_basalt.py` (`DirectFile`) + `boot.sh` |

---

### Stage 0：开机探测高速本地盘（零风险、先做）

- **改动位置**：[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)
- **做法**：在运行 `basalt_jobs.py --plan` 前，探测 VM 是否存在独立高速本地 NVMe（如 `/dev/nvme*n1` 未挂载分区或独立高速 mount，实测 `dd` 128 MiB 写速 > 600 MiB/s）。若存在，则将 `SSD_DIR` 从 `/content` 切至该盘（替代 288 MiB/s 的 `loop0`）；若无则无感回退 `/content`。

---

### Stage 1（核心）：Loader 驱动的微波分块流式加载（`~20 min → ~5.0–5.3 min`）

#### 关键源码级不变量（已核实 `loader.cpp` / `model.cpp` / `ram_load.cpp`）

1. **Wave 2 必须“按偏移排序 + ~16–20 GiB 微波分块”防 OOM 与防误打洞**：
   - `alloc_ram()` 走 `posix_memalign`/`mmap`（惰性虚拟分配），且 `pin_landed_complement()`（`plan.steps()`）和 `rounds[1]` 均在 Wave 2 全部结束后运行。
   - 若 Wave 2 整波拉 `~108 GiB` 再拷入 `alloc_ram()`，峰值达 `108 + 108 + 9 + 4 = 229 GiB > 176 GiB` 必爆 OOM；拆为 **按文件偏移升序排序的 ~16–20 GiB 微波**（拉 20 GiB → 拷入 RAM → 立刻 punch 20 GiB）后，峰值降至 **`~141 GiB`**（留足 > 35 GiB 余量），且升序排序保证相邻微波字节区间单调不相交（inward 4K punch 数学上绝不破坏后批数据）并最大化 HTTP Range 连续性（3.45 GB/s）。
2. **必须严格裁剪在 `experts` 小节内，严禁覆盖原地 Abliteration**：
   - `rounds[0]` 同时包含 `dense`/`gguf` 与 VRAM 专家；导出 `dev_spans`/`host_spans` 给 `BASALT_WAVE_FETCH_CMD` 时**必须严格限定在 `experts` 小节 `[experts_off, experts_end)` 内**，避免 Wave 1 从 HF 重拉 `dense` 覆盖已打好的 62/62 abliteration 权重。
3. **为何保持 tmpfs 微波中转而非 C++ 内置 HTTP**：
   - `108 GiB` 在 DDR5 60+ GB/s 上多一次拷贝仅耗 **~1.8s**（<0.7% 启动时间），却换来 `basalt` 零新增 `libcurl`/`OpenSSL` 依赖、复用 3.47 GB/s 零 GIL `pfetch.py`、C++ 补丁控制在 ~30 行。

```mermaid
sequenceDiagram
    participant Boot as boot.sh
    participant SSD as /content SSD (288 MiB/s)
    participant RAM as /mnt/ram tmpfs (3.45 GB/s)
    participant Loader as basalt Loader (POST /load)
    participant GPU as 96 GiB VRAM + Pinned RAM

    Note over Boot,SSD: t = 2s: 启动后台流式写入 83.93 GiB 稀疏 Engram SSD 段 (~298s, 全程并行!)
    Boot->>SSD: pfetch.py 稀疏 engram_tail.bin (Layer1 2-gram + Layer14, 两端+4KiB边界保护)

    Note over Boot,RAM: t = 2s: 仅下载非 experts 小节 (~8-10 GiB: header+meta+gguf+dspark+vision)
    Boot->>RAM: pfetch.py sparse model.basalt (~3s, experts 保持 0-block 空洞)
    Boot->>RAM: 原地 abliterate (62 个 dense 张量全在 gguf 内, ~1.5s)
    Boot->>Loader: POST /load (BASALT_ENGRAM_LAZY_OPEN=1: 先分配 4 GiB RowCache, 延迟 mmap)

    Note over Loader,GPU: Wave 1 (VRAM 波: 仅裁剪 experts 区间, 绝不碰 dense!)
    Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 dev_spans[experts] (~26s)
    Loader->>GPU: run_wave(rounds[0]) -> 写入 96 GiB VRAM (~6s)
    Loader->>RAM: punch_spans(dev_spans) -> /mnt/ram 立刻回落至 ~2 GiB

    Note over Loader,GPU: Wave 2 (RAM 微波循环: 升序 5-6 批 × ~20 GiB, 峰值 RAM <= 141 GiB!)
    loop 每批 ~20 GiB 升序 host_spans[experts]
        Loader->>RAM: BASALT_WAVE_HOOK -> pfetch.py 拉取 sub_spans (~6s)
        Loader->>GPU: run_wave(sub_wave) -> 写入 MADV_HUGEPAGE Pinned RAM (~1.5s)
        Loader->>RAM: punch_spans(sub_spans) -> 立刻释回 20 GiB tmpfs
    end

    Note over Boot,RAM: t ≈ 90s: 全部 384/384 专家已就绪! 读真实 MemAvailable 安全定容 RAM Window
    Boot->>RAM: pfetch.py 41.96 GiB Layer1 3/4-gram RAM Window (~13s, 保底留 >=10 GiB 安全余量)
    Boot->>SSD: wait PID_TAIL (Stage 1 等后台写完; Stage 2 直接开闸触发 Lazy Open!)
```

#### 具体代码改动清单

1. **[`basalt_colab/vm/basalt_jobs.py`](../basalt_colab/vm/basalt_jobs.py)**
   - **不再靠 `MemTotal` 静态盲猜硬扣双重预留**：配合 `BASALT_ENGRAM_LAZY_OPEN=1`，`/load` 时不预设过紧的 `RAM_BUDGET`（让 `expert_tier.cpp` 吃下全部剩余专家，0 专家落盘）；`/load` 完成后直接读真实 `/proc/meminfo` `MemAvailable`，按 `min(目标窗口, MemAvailable - 5 GiB)` 安全定容 RAM Window（若仍需静态预估则须按 `ram_needed + 12–13 GiB + RowCache` 计算，绝不可只扣 6–8 GiB）。
   - 新增规划模式 `wave+hybrid`（当 `bb_gib > ram_total_gib - 12.0` 且 `ram_needed_gib <= max_ram_experts_gib` 时启用，替代慢速 `ssd+hybrid`）：
     - `model_path = "/mnt/ram/model.basalt"`，`punch_file = "/mnt/ram/model.basalt"`，`parallel_tail = True`（0 字节 Backbone 写 `/content`，Engram SSD 段在 `t=2s` 立刻启动！）。
     - `j_basalt_bb_base.json`：仅包含 `header + meta + gguf + dspark/gguf + vision/mmproj`（排除 `experts` 与 `engram`，共 ~8–10 GiB），`dest_size` 仍为完整 `file_bytes`（稀疏空洞文件）。
     - 输出 **64 MiB 对齐的 RAM Window `[ram_lo, ram_hi)`**（`j_basalt_eng.json`）与 **逻辑全长稀疏 SSD 段**（`j_basalt_eng_tail.json`，两端各加 `+4096 B` 页边界防零读保护，参见第 4.3 节）。
2. **[`basalt_colab/vm/patch_basalt.py`](../basalt_colab/vm/patch_basalt.py)**
   - 新增 Patch `#9 BASALT_COLAB_WAVE_FETCH`（位于 `src/core/loader.cpp`，紧贴现有 Patch `#3b`）：
     - 检查 `BASALT_WAVE_FETCH_CMD`、`BASALT_WAVE_CHUNK_GIB`（默认 `20`）与 `BASALT_EXPERTS_RANGE`（`start:end`，严格裁剪只拉 `experts` 区间，绝不覆写 abliterated `dense`）。
     - Wave 1：导出裁剪后的 `dev_spans` → 执行 `BASALT_WAVE_FETCH_CMD <spans.json> wave1` → `run_wave` → `punch_spans`。
     - Wave 2：将 `punch_host_wave` 按**文件起始偏移升序排序**并按 `BASALT_WAVE_CHUNK_GIB`（~20 GiB）切分为若干 `sub_wave`，每批严格执行 `fetch(sub_spans)` → `run_wave(sub_wave)` → `punch_spans(sub_spans)`。
   - 新增 **`BASALT_ENGRAM_LAZY_OPEN=1`**（`src/ngram/ple_reader.cpp`）：
     - `DsModel::load()` 在 `POST /load` 最早阶段调用 `engram_->open()` 时，**仅分配 `RowCache`**（让随后的 `expert_tier.cpp` 准确扣除 `RowCache` 后再规划专家），把 `DirectFile::open()` + `mmap()` 延迟至首次 `issue()`（即 `boot.sh` Warmup 请求，此时 `/mnt/ram/engram.bin` 已按真实 `MemAvailable` 下完定型）；后台 `worker()` 在 `!m.file.is_open()` 时跳过 `keepalive` 读盘。
3. **[`basalt_colab/vm/wave_fetch.sh`](../basalt_colab/vm/)（新增 ~20 行脚本）**
   - 读取 `spans.json`，生成 `pfetch.py` job 列表（按 `src_off = dest_off = span.start, size = span.end - span.start` 直接从 `MODEL_URL` 拉取到 `/mnt/ram/model.basalt`），以 `--workers $WORKERS --chunk-mb 128` 满速写入。
4. **[`basalt_colab/vm/boot.sh`](../basalt_colab/vm/boot.sh)**
   - `wave+hybrid` 下设置 `BASALT_WAVE_FETCH_CMD` + `BASALT_EXPERTS_RANGE` + `BASALT_ENGRAM_LAZY_OPEN=1`：在 `t=2s` 启动 `PID_TAIL` 并行下 100% 必落盘的稀疏 Engram SSD 段（Layer 1 `2-gram` + Layer 14 = `83.93 GiB`）→ 前置拉 `j_basalt_bb_base.json`（~3s）→ 原地 abliteration（~1.5s）→ `POST /load`（Wave 1 + Wave 2 微波）→ 读真实 `MemAvailable` 定容并拉取 `41.96 GiB` RAM Window（~13s）→ `wait $PID_TAIL` → Warmup（触发 Lazy Open）→ 开闸。

#### 收益与风险评估
- **全量就绪时间**：`~20 s` (Colab 分配) + `max(105 s [Wave1+Wave2微波+RAM Window], 298 s [83.93 GiB 稀疏 SSD 段])` + `5 s` (warmup) ≈ **`~5.0–5.3 分钟`**（比 20 分钟提速 **3.8×**，冷启 CU 从 ~3.0 降至 **~0.77 CU**）。
- **稳定性**：100% 本地闭环，加载峰值 RAM ≤ 141 GiB、运行期保留 `~10–13 GiB` 安全余量，绝不 OOM、绝不覆写 abliteration、跨页零精度损坏。

---

### Stage 2（极致体验）：读优先异步落盘 + 无损 HTTP Range 回源（`~5.2 min → ~2.2 min` 首字可用）

在 Stage 1 基础上，`t ≈ 115 s` 时**全部 384/384 专家 + `41.96 GiB` Layer 1 3/4-gram RAM Window + `4 GiB` RowCache 均已在 VRAM/RAM 就绪**，唯一在等的是后台还要写 ~3 分钟的 `83.93 GiB` SSD Engram 段。

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

1. **单区间 RAM Window `[ram_lo, ram_hi)` + 逻辑全长稀疏 SSD 文件（含 4 KiB 跨页防零读保护）**
   - `/content/engram_tail.bin` 创建为逻辑全长（`125.89 GiB`）的**稀疏文件**，只下载不在 RAM Window 里的区间（RAM 区间保持 0-block 空洞，0 字节重复落盘）；
   - `/mnt/ram/engram.bin` 紧致存放 `[ram_lo, ram_hi)`（强制对齐 `64 MiB`，通过环境变量 `BASALT_ENGRAM_RAM_OFF` 指定起点，默认 `0` 保持 100% 向下兼容）：
     - `PleReader` 命中 `at >= ram_lo && at + rb <= ram_hi` → `mmap_head[at - ram_lo]`（80 ns 零拷贝）；
     - 未命中 → `DirectFile` 直接 `pread(fd2, buf, len, at)`（**无需再做 `offset - split_bytes` 偏移换算**！）。
   - **关键边界防坑（防静默读出全零）**：因 `Q5_K` 行宽 `176 B` 不能整除 `4096 B`，跨 `ram_lo`/`ram_hi` 边界的那两行会触发 `DirectFile` 4 KiB 读进入空洞；**稀疏 SSD 文件下载区间在 `ram_lo` 之后与 `ram_hi` 之前各多覆盖 1 页（`+4096 B`，总共仅多占 8 KiB）**，保证全表 `768,022,850` 行 100% 逐比特精确。
   - **放置策略**：
     - 物理计算表明 G4 扣除 `~108 GiB` 专家 + `~9 GiB` 系统/残余 + `4 GiB` `RowCache` + `≥5 GiB` 动态缓冲后仅剩 `~50 GiB < 62.94 GiB`（无法安全硬塞完整 Layer 1）；
     - 因此固定优先锁定 **`[20.98 GiB, 62.94 GiB)`（对齐后 `~41.96 GiB`，覆盖 100% Layer 1 `3-gram + 4-gram`）** 进 `/mnt/ram`，仅将最易命中的 Layer 1 `2-gram`（`20.98 GiB`）与 Layer 14（`62.95 GiB`）放 `/content`；
     - **顺带惠及 `UD-IQ2_XXS`**：`UD-IQ2_XXS` 仅需 `~77 GiB` RAM 专家，套用同一策略可将 **Layer 1 全部（`51.50 GiB`）+ Layer 14 `3-gram + 4-gram`（`34.34 GiB`）= `85.84 GiB`** 全部装进 RAM，落盘的仅剩 `17.17 GiB` Layer 14 `2-gram`（有效盘读砍掉 ~80–90%）。
2. **`4 GiB` `RowCache` + `RowCache::insert()` 统一防 4-gram 冲刷过滤（拒绝 `PageCache`）**
   - **保持 `/content` 走 `O_DIRECT`（0 `PageCache`）**：`PageCache` 以 4096 B 页缓存 176 B 均匀哈希行，空间浪费高达 **95.7%（密度比 `RowCache` 低 23×）**，且易与 `Pinned RAM`/`tmpfs` 争抢内存触发 Colab cgroup direct reclaim 或 OOM Kill。
   - **为何 `4 GiB` 就是真实甜点（无需扩大 Patch `#7` 现有 `4096 MiB` clamp）**：
     - 即使一场长达 **50 万 token** 的多轮编码会话，出现过的不同 2-gram 总数也仅约 5–20 万个（× 8 heads ≈ `160 万` 行 × `176 B` ≈ **`281 MiB`**）；
     - `4 GiB` `RowCache`（`2,440 万` 行，等效 `92 GiB PageCache`）已足够将 Layer 1 + Layer 14 的高频 2-gram 吃到 **95–99% 命中率**（Layer 1 有效读盘趋近于 `< 0.3 次/token`），且不会在 `POST /load` 早期抢占 `expert_tier.cpp` 内存预算。
   - **5 行防冲刷准入过滤（必须放在 `RowCache::insert()` 内部唯一入口）**：`ple_reader.cpp` 在 `worker()`（单页行）和 `collect()`（跨页 `split` 行）两处均调用 `m.cache.insert()`；在 `RowCache::insert()` 内部直接跳过落在 SSD 的 **Layer 14 `4-gram`** 行号区间，彻底防止 16K Prefill 一次性冷 4-gram 冲刷热 2/3-gram 缓存。
3. **Colab G4 176 GiB 黄金安全配比（`UD-Q2_K_XL`）**
   - **Pinned RAM 专家**：`~106–110 GiB`（第一优先级，0 专家落盘）
   - **OS / CUDA / `model.basalt` 残余 / host buffers**：`~9 GiB`（刚性底噪）
   - **`RowCache`**：**`4 GiB`**（`POST /load` 早期分配，Patch `#7` 现有上限直接支持）
   - **静态 `/mnt/ram` RAM Window (`mmap`)**：**`~42 GiB`**（精确覆盖 Layer 1 `3-gram + 4-gram` `[20.98, 62.94)`，`/load` 后按真实 `MemAvailable - 5 GiB` 安全定容）
   - **动态运行期余量**：**`~10–13 GiB`**（充裕覆盖多图 Vision `dynamic`、1M FP8 KV、并发请求与 `1.2 GiB` 熔断线，绝不 OOM）
   - **`PageCache`**：`0`（保持 `O_DIRECT`）
4. **放开 `PleReader` `max_inflight`（64 → 256）+ Stage 0 探测本地 NVMe**
   - 在 [`patch_basalt.py`](../basalt_colab/vm/patch_basalt.py) 中将 `engram_->open(..., 64, ...)` 改为可由 `BASALT_ENGRAM_INFLIGHT` 配置（默认 `256`，匹配 `DirectFile` 256 线程），提升 16K Prefill 并发队列深度；若 Stage 0 命中本地 NVMe 则 IOPS 瓶颈直接归零。
5. **备选（自转混合包，几乎无损）**：`UD-Q2_K_XL` 专家 + 4-bit Engram（`MXFP4` 136 B/行共 `97.3 GiB`，**Layer 1 仅 `48.6 GiB`**；或 `IQ4_NL` 共 `103 GiB`），立省 **23–29 GiB RAM**，`parallel=2` 下也能 100% 装下整个 Layer 1 + 部分 Layer 14。

### 4.4 客观物理边界与残余局限（无法靠软件消除）

1. **冷启动写盘下限**：无本地 NVMe 时，~83.9 GiB 稀疏 Engram 写入 288 MiB/s 的 `/content` 存在 ~298s 的硬除法下限（Stage 1 约 `~5.0–5.3 min`）；压至 `~2.2 min` 需启用 Stage 2（前 ~3 分钟冷行偶发 5–15 ms HTTP 回源）。
2. **首次冷 16K–128K 超长 Prefill 的 Layer 14 读盘延迟**：Layer 14 主体在 `/content`，首轮超长上下文 Prefill 仍受 GCE PD 随机读 IOPS 限制；第二轮起由 32 GiB `conv_cache` 增量复用 + `4 GiB RowCache` 恢复满速。
3. **Colab 无 Swap 约束**：严守 `/load` 后 `MemAvailable` 至少保留 `≥ 5 GiB` 安全垫，不试图把 62.94 GiB 全量 Layer 1 硬塞进 RAM。
4. **上游补丁维护面**：外挂补丁集中在 `loader.cpp` / `ple_reader.cpp` / `direct_file.cpp` / `model.cpp` 4 个文件；每次升级 Basalt tag 均由 CI 锚点检查与预编译 `sm_120` 流水线把关。

---

## 5. 升级与验证步骤（待上游就绪后按序执行）

1. **升级 Basalt 基线**：
   - 将 [`manifest.json`](../basalt_colab/vm/manifest.json) 指向新 tag（`v0.5.8`）及对应 commit；
   - 在本地或 CI 验证 `patch_basalt.py` 对新 tag 全部现有 patch 锚点 100% 匹配；
   - 触发 `.github/workflows/build-basalt-prebuilt.yml` 发布 `prebuilt-v0.5.8` 并锁定 `prebuilt_sha256`。
2. **接入 `UD-Q2_K_XL` 包直链**：
   - 回填 `manifest.json` `packs.ud-q2_k_xl.url`；
   - 放行 `config.py` 对 `ud-q2_k_xl` 免填 `--pack-url`（沿用 `manifest.json` 内置 URL）。
3. **先落地 Stage 0 + Stage 1（Micro-Wave）+ Engram 推理优化（`LAZY_OPEN` + RAM Window + `+4096 B` 边界保护 + `4 GiB` 过滤 `RowCache` + `max_inflight=256`），单次真机 G4 实测验收**：
   - 核验 `timeline.txt`：Wave 1（仅 `experts` 不覆写 abliteration）/ Wave 2 升序微波耗时与 punch 回收、整体内存峰值（目标 ≤ 141 GiB、绝不 OOM）、`/content` 写入总量（0 字节 Backbone 落盘、仅 ~83.9 GiB 稀疏 Engram）；
   - 核验 `basalt` 启动日志：`expert homes`（目标 143–153/384 in VRAM，0 on SSD）、`0 lossy flags`、abliteration 生效验证、RAM Window 精确覆盖 Layer 1 `3-gram+4-gram`（`41.96 GiB`）且边界两行逐字节一致；
   - 核验推理吞吐与 Engram I/O：实测 `conv / struct / prose` Decode 吞吐、16K 长 Prompt Prefill 吞吐以及 `PleReader` 读盘延迟（p50/p99）与 `RowCache` 命中率。
4. **按需叠加 Stage 2（读优先异步落盘）**：
   - 若希望将 ~5.2 min 进一步压至 ~2.2 min，再实现 `ready_bitmap` + Unix-socket Keep-Alive Range 回源 + `$R/READY` 后的 `ionice -c 3` 限速退让，并用 `test_config.py` 本地单测验证未就绪 chunk 回源、就绪切换与稀疏文件边界页的逐字节一致性。
