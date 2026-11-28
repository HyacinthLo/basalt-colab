#!/bin/bash
# Cold start on Colab G4 using Basalt v0.6.0 (DeepSeek-V4.1-Flash):
#   0. Stage 0: probe for fast local NVMe scratch (>600 MiB/s synced write, >=120 GiB free); fallback /content
#   1. Inspect 64 KiB .basalt header (+ meta for hybrid) -> auto-plan storage tier:
#        - ram+ram     (UD-IQ1_M):   100% RAM backbone + 100% RAM Engram
#        - ram+hybrid  (UD-IQ2_XXS): RAM backbone + lazy multi-window RAM (~83-86 GiB) + sparse SSD tail (~17-20 GiB, parallel at t=2s)
#        - wave+hybrid (UD-Q2_K_XL): Stage 1 micro-wave zero-SSD backbone (Wave 1 VRAM + Wave 2 ~20 GiB chunks) + lazy RAM window + sparse SSD (~84 GiB, parallel at t=2s)
#   2. Parallel: fetch/build patched Basalt + fetch sparse backbone (or base sections in wave+*) + fetch 43 KB abliteration edit
#                (+ parallel sparse SSD Engram tail download when PARALLEL_TAIL=1)
#   3. In-place abliteration via standalone abliterate (~1.5-3s, 62/62 tensors, tunable UNCENSOR_SCALE)
#   4. Start basalt serve + POST /load (punch-on-read + wave_fetch.sh in wave+*; hybrid allocates 4 GiB RowCache & defers Engram open)
#   5. Finalize RAM Engram windows from real post-/load MemAvailable (--finalize-engram) & join SSD tail
#   6. Warmup (triggers lazy multi-window mmap) + open gate.py on :18080 + arm VM-side hard-cap backstop
set -uo pipefail
: "${CTX:?}"
export PORT=18081 UNCENSOR=${UNCENSOR:-1} UNCENSOR_SCALE=${UNCENSOR_SCALE:-1.0} WORKERS=${WORKERS:-64}
export PARALLEL=${PARALLEL:-2} KV_TYPE=${KV_TYPE:-fp8} VISION_MODE=${VISION_MODE:-dynamic}
export SPEC_MAX=${SPEC_MAX:-6} DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
export VRAM_RESERVE=${VRAM_RESERVE:-450} RAM_BUDGET=${RAM_BUDGET:-0}
export ROUTING_MODE=${ROUTING_MODE:-exact} POOL_WORKERS=${POOL_WORKERS:-0} ENGRAM_CACHE_MIB=${ENGRAM_CACHE_MIB:-4096}
export ENGRAM_INFLIGHT=${ENGRAM_INFLIGHT:-256}
export CONV_CACHE_MIB=${CONV_CACHE_MIB:-32768}
export MODEL_QUANT=${MODEL_QUANT:-ud-iq1_m}
export MODEL_ALIAS=${MODEL_ALIAS:-DeepSeek-V4.1-Flash-1017-UD-IQ1_M}
export MODEL_URL=${MODEL_URL:-}
[ -s /content/secrets/hf_hdr ] && export HF_HDR=/content/secrets/hf_hdr
R=/content/res; V=/content/vm; D=/content/basalt
mkdir -p $R /mnt/ram /content/patch $D
rm -rf $R/READY $R/load_done $R/engram_claim

T0=$(date +%s)
export BOOT_T0="$T0"
st() { echo "$(( $(date +%s) - T0 ))s $(date +%T) $*" >> $R/timeline.txt; }
seen() { grep -qE "$1" $R/timeline.txt; }
hdr() { echo "Authorization: Bearer $(cat /content/secrets/key)"; }

# Read pinned URLs & SHA-256 from $V/manifest.json
eval "$(python3 - "$V/manifest.json" "$MODEL_QUANT" <<'EOF'
import json, shlex, sys
m = json.load(open(sys.argv[1]))
q = sys.argv[2].strip().lower()
packs = m.get("packs") or {}
pack_url = (packs.get(q) or {}).get("url") or (m.get("pack_url", "") if q in ("ud-iq1_m", "iq1_m") else "")
for env_k, val in (
    ("ABLIT_URL", m.get("abliteration_url", "")),
    ("ABLIT_SHA", m.get("abliteration_sha256", "")),
    ("MANIFEST_PREBUILT_URL", m.get("prebuilt_url", "")),
    ("MANIFEST_PREBUILT_SHA", m.get("prebuilt_sha256", "")),
    ("MANIFEST_PACK_URL", pack_url),
):
    print(f"export {env_k}={shlex.quote(str(val))}")
EOF
)"

[ -n "$MODEL_URL" ] || MODEL_URL="$MANIFEST_PACK_URL"
if [ -z "$MODEL_URL" ]; then
  st "MODEL_URL missing for quant=$MODEL_QUANT FAILED"
  exit 1
fi

st "boot start engine=basalt quant=$MODEL_QUANT model=$MODEL_ALIAS ref=${BASALT_REF:-pinned} ctx=$CTX parallel=$PARALLEL kv=$KV_TYPE vision=$VISION_MODE route=$ROUTING_MODE vram_res=$VRAM_RESERVE uncensored=$UNCENSOR scale=$UNCENSOR_SCALE"

# Stage 0: Probe for fast local NVMe scratch mount (>600 MiB/s synced write, >=120 GiB free); fallback to /content
SSD_DIR=/content
if [ -n "${BASALT_SSD_DIR:-}" ]; then
  SSD_DIR="$BASALT_SSD_DIR"
  mkdir -p "$SSD_DIR"
else
  while read -r dev mnt fstype _; do
    case "$mnt" in
      /|/boot*|/content|/mnt/ram|/proc*|/sys*|/dev*|/run*|/tmp) continue ;;
    esac
    case "$fstype" in
      ext4|xfs|btrfs) ;;
      *) continue ;;
    esac
    [ -w "$mnt" ] || continue
    avail_g=$(df -BG --output=avail "$mnt" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${avail_g:-0}" -ge 120 ] || continue
    probe="$mnt/.basalt_io_probe.$$"
    w_mibs=$(python3 - "$probe" <<'PYEOF'
import os, sys, time
p = sys.argv[1]
sz = 128 * 1024 * 1024
try:
    t0 = time.monotonic()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        buf = b"\x5a" * (4 * 1024 * 1024)
        for _ in range(sz // len(buf)):
            os.write(fd, buf)
        os.fsync(fd)
    finally:
        os.close(fd)
    dt = max(time.monotonic() - t0, 1e-4)
    print(int((sz / 1048576.0) / dt))
except OSError:
    print(0)
finally:
    try:
        os.unlink(p)
    except OSError:
        pass
PYEOF
    )
    if [ "${w_mibs:-0}" -gt 600 ]; then
      SSD_DIR="$mnt/basalt_scratch"
      mkdir -p "$SSD_DIR"
      st "stage0: selected fast local scratch $SSD_DIR (dev=$dev fs=$fstype free=${avail_g}GiB write=${w_mibs}MiB/s)"
      break
    fi
  done < /proc/mounts
fi
export SSD_DIR
export CONV_CACHE_DIR="${CONV_CACHE_DIR:-$SSD_DIR/conv_cache}"

RAM_GIB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
TMPFS_GIB=$(awk '/MemTotal/ {print int($2/1048576) - 6}' /proc/meminfo)
DISK_FREE_GIB=$(df -BG --output=avail "$SSD_DIR" | tail -1 | tr -dc '0-9')
VRAM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", ($1>0?$1:98304)/1024}')
VRAM_GIB=${VRAM_GIB:-96.0}
mountpoint -q /mnt/ram || mount -t tmpfs -o size=${TMPFS_GIB}G tmpfs /mnt/ram

# Inspect remote 64 KiB .basalt header and plan ram+ram / ram+hybrid / ssd+hybrid (fail-fast if > VM RAM+VRAM)
PLAN_OUT=$(python3 $V/basalt_jobs.py --plan --quant "$MODEL_QUANT" --url "$MODEL_URL" \
  --ssd-dir "$SSD_DIR" --ram-total-gib "$RAM_GIB" --disk-free-gib "$DISK_FREE_GIB" --vram-gib "$VRAM_GIB" \
  --ram-budget "$RAM_BUDGET" 2> $R/plan_err.txt) || {
  st "storage plan FAILED ($(tr '\n' ' ' < $R/plan_err.txt))"
  exit 1
}
eval "$PLAN_OUT"
ENG_SPLIT_DESC="${ENGRAM_GIB}GiB"
if [ -n "${ENGRAM_TAIL_PATH:-}" ]; then
  ENG_SPLIT_DESC="$(python3 -c "print(f'{int($ENGRAM_HEAD_BYTES)/1073741824:.1f}GiB-RAM+{int($ENGRAM_TAIL_BYTES)/1073741824:.1f}GiB-SSD(lazy)')")"
fi
st "storage plan mode=$STAGE_MODE bb=${BACKBONE_GIB}GiB eng=${ENG_SPLIT_DESC} ram_budget=${RAM_BUDGET}GiB parallel_tail=${PARALLEL_TAIL} ssd_dir=$SSD_DIR (vm ram=${RAM_GIB}GiB disk=${DISK_FREE_GIB}GiB vram=${VRAM_GIB}GiB)"

# Probe hole-punching on $SSD_DIR before downloading 200+ GiB in ssd+* modes
if [[ "$STAGE_MODE" == ssd+* ]]; then
  python3 $V/basalt_jobs.py --test-punch "$SSD_DIR" > $R/punch_probe.txt 2>&1 || {
    st "GUARD: $SSD_DIR does not support FALLOC_FL_PUNCH_HOLE ($(cat $R/punch_probe.txt 2>/dev/null)) FAILED"
    exit 1
  }
  st "punch probe $SSD_DIR ok"
fi

# Stage 1 Micro-Wave: stream experts on demand in /mnt/ram/model.basalt (0 bytes backbone on SSD)
WAVE_JOB_ARGS=()
if [[ "$STAGE_MODE" == wave+* ]] && [ -n "${EXPERTS_RANGE:-}" ]; then
  export BASALT_WAVE_FETCH_CMD="$V/wave_fetch.sh"
  export BASALT_EXPERTS_RANGE="$EXPERTS_RANGE"
  export BASALT_WAVE_CHUNK_GIB="${WAVE_CHUNK_GIB:-20}"
  export MODEL_FILE_BYTES
  WAVE_JOB_ARGS=(--exclude-experts)
fi

if [ -n "${ENGRAM_TAIL_PATH:-}" ]; then
  # Hybrid modes: allocate RowCache during /load, defer Engram mmap/open until post-/load real MemAvailable finalize
  export BASALT_ENGRAM_LAZY_OPEN=1
  export BASALT_ENGRAM_LAYOUT=/mnt/ram/engram_layout.txt
  rm -f "$BASALT_ENGRAM_LAYOUT" /mnt/ram/engram_*.bin "$R/j_basalt_eng_tail.json" "$R/j_basalt_eng_tail_extra.json"
  EST_RAM_AVAIL_GIB=$(python3 -c "print(max(0.0, float('$RAM_GIB') - float('${RAM_NEEDED_GIB:-0}') - 11.5 - float('$ENGRAM_CACHE_MIB')/1024.0))")
  PLAN_EXTRA_ARGS=(--engram-plan-out "$R/engram_plan.json" --est-ram-avail-gib "$EST_RAM_AVAIL_GIB")
  [ "${PARALLEL_TAIL:-0}" = 0 ] && PLAN_EXTRA_ARGS+=(--no-initial-ssd)
  python3 $V/basalt_jobs.py --quant "$MODEL_QUANT" --url "$MODEL_URL" \
    --backbone-dest "$MODEL_PATH" --backbone-jobs "$R/j_basalt_bb.json" \
    --engram-dest "$ENGRAM_PATH" --engram-jobs "$R/j_basalt_eng.json" \
    --engram-tail-dest "$ENGRAM_TAIL_PATH" --engram-head-bytes "${ENGRAM_HEAD_BYTES:-0}" \
    --engram-tail-jobs "$R/j_basalt_eng_tail.json" \
    --engram-size-out "$R/engram_size.txt" "${WAVE_JOB_ARGS[@]}" "${PLAN_EXTRA_ARGS[@]}" > $R/fetch.log 2>&1 || { st "basalt_jobs FAILED"; exit 1; }
  ENGRAM_TOTAL_SIZE=$(cat $R/engram_size.txt)
  # Pre-create logical full-size 0-block sparse SSD file for parallel/delta range writes
  python3 -I -c "
import os
f = os.open('$ENGRAM_TAIL_PATH', os.O_WRONLY | os.O_CREAT, 0o644)
os.ftruncate(f, int('$ENGRAM_TOTAL_SIZE'))
os.close(f)
"
else
  python3 $V/basalt_jobs.py --quant "$MODEL_QUANT" --url "$MODEL_URL" \
    --backbone-dest "$MODEL_PATH" --backbone-jobs "$R/j_basalt_bb.json" \
    --engram-dest "$ENGRAM_PATH" --engram-jobs "$R/j_basalt_eng.json" \
    --engram-size-out "$R/engram_size.txt" "${WAVE_JOB_ARGS[@]}" > $R/fetch.log 2>&1 || { st "basalt_jobs FAILED"; exit 1; }
  ENGRAM_HEAD_SIZE=$(cat $R/engram_size.txt)
  # Pre-create 0-block sparse Engram file so PleReader::open sees exact logical size during /load
  python3 -I -c "
import os
f = os.open('$ENGRAM_PATH', os.O_WRONLY | os.O_CREAT, 0o644)
os.ftruncate(f, int('$ENGRAM_HEAD_SIZE'))
os.close(f)
"
fi

# Launch parallel build + abliteration fetch + backbone fetch (+ SSD Engram tail if PARALLEL_TAIL=1!)
SSD_WORKERS=$(( WORKERS < 16 ? WORKERS : 16 ))
PID_TAIL=""
if [ "${PARALLEL_TAIL:-0}" = 1 ]; then
  if [ -n "${ENGRAM_TAIL_PATH:-}" ] && [ -s "$R/j_basalt_eng_tail.json" ] && grep -q '"url"' "$R/j_basalt_eng_tail.json"; then
    { python3 $V/pfetch.py $R/j_basalt_eng_tail.json --workers $SSD_WORKERS --chunk-mb 64 --tag basalt-eng-tail --log $R/fetch_tail.log \
      && st "engram tail done ($ENGRAM_TAIL_PATH)" || st "engram tail FAILED"; } &
    PID_TAIL=$!
  elif [ "$STAGE_MODE" = "ram+ssd" ]; then
    { python3 $V/pfetch.py $R/j_basalt_eng.json --workers $SSD_WORKERS --chunk-mb 64 --tag basalt-eng --log $R/fetch_tail.log \
      && st "engram done ($ENGRAM_PATH)" || st "engram FAILED"; } &
    PID_TAIL=$!
  fi
fi

{ bash $V/build.sh > $R/basalt_build.log 2>&1 && st "basalt built" || st "basalt build FAILED ($(tail -1 $R/build.txt 2>/dev/null))"; } &
PID_BUILD=$!
PID_ABLIT=""
if [ "$UNCENSOR" = 1 ]; then
  { curl -sfL --retry 5 ${HF_HDR:+-H @$HF_HDR} -o /content/patch/abliteration-edit.npz "$ABLIT_URL" \
    && sha256sum /content/patch/abliteration-edit.npz | grep -q "^$ABLIT_SHA" \
    && st "abliteration edit fetched" || st "abliteration edit FAILED"; } &
  PID_ABLIT=$!
fi
BB_WORKERS=$WORKERS
BB_CHUNK=128
[[ "$MODEL_PATH" != /mnt/ram/* ]] && { BB_WORKERS=$SSD_WORKERS; BB_CHUNK=64; }
{ python3 $V/pfetch.py $R/j_basalt_bb.json --workers $BB_WORKERS --chunk-mb $BB_CHUNK --tag basalt-bb --log $R/fetch.log \
  && st "backbone fetched ($MODEL_QUANT -> $MODEL_PATH)" || st "backbone FAILED"; } &
PID_BB=$!

# Wait ONLY for build + abliteration + backbone (let SSD Engram tail keep streaming concurrently during /load!)
wait $PID_BUILD $PID_BB ${PID_ABLIT:+$PID_ABLIT}
seen FAILED && { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "fetch/build FAILED, stopping"; exit 1; }

if [ "$UNCENSOR" = 1 ]; then
  python3 -I $V/abliterate_basalt.py "$MODEL_PATH" /content/patch/abliteration-edit.npz "$D/abliterate" \
    --scale "$UNCENSOR_SCALE" > $R/patch.log 2>&1 \
    && st "abliteration applied (scale=$UNCENSOR_SCALE)" || { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "abliteration FAILED ($(tail -1 $R/patch.log 2>/dev/null))"; exit 1; }
fi

bash $V/serve.sh
st "server started"
for k in $(seq 1 120); do
  curl -sf -o /dev/null -H "$(hdr)" http://127.0.0.1:$PORT/health && { st "basalt http ready"; break; }
  [ -d /proc/$(cat $R/server.pid) ] || { st "server died ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }
  sleep 0.5
done

# Eager weight load (with punch-on-read reclaiming backbone storage before RAM Engram download)
curl -sf --max-time 900 -X POST -H "$(hdr)" -H "Content-Type: application/json" -d '{}' \
  http://127.0.0.1:$PORT/load > $R/load.json 2>&1 && st "weights loaded" || {
    [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true
    st "weights load FAILED ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1;
  }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${VRAM:-0}" -lt 60000 ]; then
  [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true
  st "GUARD: VRAM ${VRAM} MiB after /load FAILED"; exit 1
fi
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  python3 $V/basalt_jobs.py --backbone-dest "$MODEL_PATH" --punch-sections dspark/gguf dense > $R/punch.txt 2>&1 || true
fi
st "pin: VRAM ${VRAM} MiB, model $(du -sh "$MODEL_PATH" | awk '{print $1}'), mode=$STAGE_MODE"

# Stream RAM Engram (full in ram+ram, multi-window RAM + sparse SSD in hybrid modes)
if [[ "$ENGRAM_PATH" == /mnt/ram/* ]]; then
  ( for k in $(seq 1 600); do [ -f $R/READY ] && exit 0; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      [ "$a" -lt 1200 ] && { for p in $(pgrep -f "[p]fetch.py"); do kill $p; done; st "GUARD: ${a} MiB RAM available, engram fetch stopped FAILED"; exit 0; }; sleep 2; done ) &
fi

if [ "${BASALT_ENGRAM_LAZY_OPEN:-0}" = 1 ]; then
  MEM_AVAIL_GIB=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
  FIN_MSG=$(python3 $V/basalt_jobs.py --finalize-engram \
    --plan-json "$R/engram_plan.json" \
    --mem-avail-gib "$MEM_AVAIL_GIB" \
    --ram-dir /mnt/ram \
    --ssd-file "$ENGRAM_TAIL_PATH" \
    --layout-out "$BASALT_ENGRAM_LAYOUT" \
    --ram-jobs-out "$R/j_basalt_eng.json" \
    --ssd-extra-jobs-out "$R/j_basalt_eng_tail_extra.json" 2>&1) || {
      [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true
      st "engram finalize FAILED ($FIN_MSG)"; exit 1;
    }
  st "$FIN_MSG"

  # If 100% of Engram fit in RAM windows, cancel any unnecessary background SSD tail fetch
  if ! grep -q '^ssd ' "$BASALT_ENGRAM_LAYOUT"; then
    [ -n "$PID_TAIL" ] && { kill $PID_TAIL 2>/dev/null || true; PID_TAIL=""; }
    rm -f "$ENGRAM_TAIL_PATH"
  elif [ "${PARALLEL_TAIL:-0}" = 0 ] && [ -s "$R/j_basalt_eng_tail_extra.json" ] && grep -q '"url"' "$R/j_basalt_eng_tail_extra.json"; then
    # In ssd+hybrid (PARALLEL_TAIL=0), start exact post-/load SSD ranges concurrently with RAM windows
    mv -f "$R/j_basalt_eng_tail_extra.json" "$R/j_basalt_eng_tail.json"
    { python3 $V/pfetch.py $R/j_basalt_eng_tail.json --workers $SSD_WORKERS --chunk-mb 64 --tag basalt-eng-tail --log $R/fetch_tail.log \
      && st "engram tail done ($ENGRAM_TAIL_PATH)" || st "engram tail FAILED"; } &
    PID_TAIL=$!
  fi

  if [ -s "$R/j_basalt_eng.json" ] && grep -q '"url"' "$R/j_basalt_eng.json"; then
    python3 $V/pfetch.py $R/j_basalt_eng.json --workers $WORKERS --chunk-mb 128 --tag basalt-eng --log $R/fetch.log \
      && st "engram RAM windows done" || { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "engram FAILED"; exit 1; }
  fi
  if [ -n "$PID_TAIL" ]; then
    wait $PID_TAIL
  fi
  # Fetch any delta SSD ranges if real MemAvailable was tighter than initial estimate (disjoint from initial ranges)
  if [ -s "$R/j_basalt_eng_tail_extra.json" ] && grep -q '"url"' "$R/j_basalt_eng_tail_extra.json"; then
    python3 $V/pfetch.py $R/j_basalt_eng_tail_extra.json --workers $SSD_WORKERS --chunk-mb 64 --tag basalt-eng-tail-extra --log $R/fetch_tail.log \
      && st "engram tail delta done ($ENGRAM_TAIL_PATH)" || { st "engram tail delta FAILED"; exit 1; }
  fi
else
  # Non-hybrid fast path (ram+ram UD-IQ1_M, ram+ssd, ssd+ssd)
  if [ "$STAGE_MODE" != "ram+ssd" ] || [ "${PARALLEL_TAIL:-0}" = 0 ]; then
    ENG_WORKERS=$WORKERS
    ENG_CHUNK=128
    [[ "$ENGRAM_PATH" != /mnt/ram/* ]] && { ENG_WORKERS=$SSD_WORKERS; ENG_CHUNK=64; }
    python3 $V/pfetch.py $R/j_basalt_eng.json --workers $ENG_WORKERS --chunk-mb $ENG_CHUNK --tag basalt-eng --log $R/fetch.log \
      && st "engram done ($ENGRAM_PATH)" || { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "engram FAILED"; exit 1; }
  fi
  if [ -n "$PID_TAIL" ]; then
    wait $PID_TAIL
  fi
fi

seen FAILED && { st "engram FAILED"; exit 1; }
free -b > $R/free_after_engram.txt
touch $R/READY

curl -sf --max-time 180 -H "$(hdr)" http://127.0.0.1:$PORT/v1/chat/completions -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_ALIAS\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK.\"}],\"max_tokens\":8,\"temperature\":0,\"reasoning_effort\":\"low\"}" \
  > $R/warmup.json && st "first token (warmup done)" || { st "warmup FAILED"; exit 1; }

setsid nohup python3 $V/gate.py 18080 $PORT > $R/gate.log 2>&1 < /dev/null &
for k in $(seq 1 50); do grep -q "gate open" $R/gate.log 2>/dev/null && break; sleep 0.1; done
grep -q "gate open" $R/gate.log && st "gate open 18080" || st "gate FAILED"

# VM-side hard-cap backstop if the local host sleeps or loses power
if [ -n "${MAX_HOURS:-}" ]; then
  CAP_S=$(python3 -c "print(max(600, int(float('$MAX_HOURS') * 3600) + 300))" 2>/dev/null || echo 21900)
  setsid bash -c "sleep $CAP_S; pkill -f '[g]ate.py' || true; kill \$(cat $R/server.pid 2>/dev/null) 2>/dev/null || true; rm -rf /content/secrets" > /dev/null 2>&1 < /dev/null &
fi
