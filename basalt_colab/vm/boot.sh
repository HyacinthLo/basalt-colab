#!/bin/bash
# Cold start on Colab G4 using Basalt v0.5.7 (DeepSeek-V4.1-Flash):
#   1. Inspect 64 KiB .basalt header -> auto-plan storage tier:
#        - ram+ram    (UD-IQ1_M):   100% RAM backbone + 100% RAM Engram
#        - ram+hybrid (UD-IQ2_XXS): RAM backbone + ~78 GiB RAM Engram head + ~25 GiB SSD tail (parallel at t=2s!)
#        - ssd+hybrid (UD-Q2_K_XL): SSD backbone (punch-on-read) + ~36 GiB RAM Engram head + ~67 GiB SSD tail
#   2. Parallel: fetch/build patched Basalt + fetch sparse backbone + fetch 43 KB abliteration edit
#                (+ parallel SSD Engram tail download when PARALLEL_TAIL=1)
#   3. In-place abliteration via standalone abliterate (~1.5-3s, 62/62 tensors, tunable UNCENSOR_SCALE)
#   4. Start basalt serve + POST /load (punch-on-read reclaims backbone storage before RAM Engram stream)
#   5. Stream RAM Engram head (~25-32s) & join background SSD Engram tail (usually already done!)
#   6. Warmup + open gate.py on :18080 + arm VM-side hard-cap backstop
set -uo pipefail
: "${CTX:?}"
export PORT=18081 UNCENSOR=${UNCENSOR:-1} UNCENSOR_SCALE=${UNCENSOR_SCALE:-1.0} WORKERS=${WORKERS:-64}
export PARALLEL=${PARALLEL:-2} KV_TYPE=${KV_TYPE:-fp8} VISION_MODE=${VISION_MODE:-dynamic}
export SPEC_MAX=${SPEC_MAX:-6} DRAFT_EXPERTS=${DRAFT_EXPERTS:-32}
export VRAM_RESERVE=${VRAM_RESERVE:-700} RAM_BUDGET=${RAM_BUDGET:-0}
export CONV_CACHE_MIB=${CONV_CACHE_MIB:-32768}
export MODEL_QUANT=${MODEL_QUANT:-ud-iq1_m}
export MODEL_ALIAS=${MODEL_ALIAS:-DeepSeek-V4.1-Flash-1017-UD-IQ1_M}
export MODEL_URL=${MODEL_URL:-}
[ -s /content/secrets/hf_hdr ] && export HF_HDR=/content/secrets/hf_hdr
R=/content/res; V=/content/vm; D=/content/basalt
mkdir -p $R /mnt/ram /content/patch $D
rm -f $R/READY

T0=$(date +%s)
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

st "boot start engine=basalt quant=$MODEL_QUANT model=$MODEL_ALIAS ref=${BASALT_REF:-pinned} ctx=$CTX parallel=$PARALLEL kv=$KV_TYPE vision=$VISION_MODE uncensored=$UNCENSOR scale=$UNCENSOR_SCALE"

RAM_GIB=$(awk '/MemTotal/ {printf "%.1f", $2/1048576}' /proc/meminfo)
TMPFS_GIB=$(awk '/MemTotal/ {print int($2/1048576) - 6}' /proc/meminfo)
DISK_FREE_GIB=$(df -BG --output=avail /content | tail -1 | tr -dc '0-9')
VRAM_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | awk '{printf "%.1f", ($1>0?$1:98304)/1024}')
VRAM_GIB=${VRAM_GIB:-96.0}
mountpoint -q /mnt/ram || mount -t tmpfs -o size=${TMPFS_GIB}G tmpfs /mnt/ram

# Inspect remote 64 KiB .basalt header and plan ram+ram / ram+hybrid / ssd+hybrid (fail-fast if > VM RAM+VRAM)
PLAN_OUT=$(python3 $V/basalt_jobs.py --plan --quant "$MODEL_QUANT" --url "$MODEL_URL" \
  --ram-total-gib "$RAM_GIB" --disk-free-gib "$DISK_FREE_GIB" --vram-gib "$VRAM_GIB" \
  --ram-budget "$RAM_BUDGET" 2> $R/plan_err.txt) || {
  st "storage plan FAILED ($(tr '\n' ' ' < $R/plan_err.txt))"
  exit 1
}
eval "$PLAN_OUT"
ENG_SPLIT_DESC="${ENGRAM_GIB}GiB"
if [ -n "${ENGRAM_TAIL_PATH:-}" ]; then
  ENG_SPLIT_DESC="$(python3 -c "print(f'{int($ENGRAM_HEAD_BYTES)/1073741824:.1f}GiB-RAM+{int($ENGRAM_TAIL_BYTES)/1073741824:.1f}GiB-SSD')")"
fi
st "storage plan mode=$STAGE_MODE bb=${BACKBONE_GIB}GiB eng=${ENG_SPLIT_DESC} ram_budget=${RAM_BUDGET}GiB parallel_tail=${PARALLEL_TAIL} (vm ram=${RAM_GIB}GiB disk=${DISK_FREE_GIB}GiB vram=${VRAM_GIB}GiB)"

# Probe hole-punching on /content before downloading 200+ GiB in ssd+* modes
if [[ "$STAGE_MODE" == ssd+* ]]; then
  python3 $V/basalt_jobs.py --test-punch /content > $R/punch_probe.txt 2>&1 || {
    st "GUARD: /content does not support FALLOC_FL_PUNCH_HOLE ($(cat $R/punch_probe.txt 2>/dev/null)) FAILED"
    exit 1
  }
  st "punch probe /content ok"
fi

python3 $V/basalt_jobs.py --quant "$MODEL_QUANT" --url "$MODEL_URL" \
  --backbone-dest "$MODEL_PATH" --backbone-jobs "$R/j_basalt_bb.json" \
  --engram-dest "$ENGRAM_PATH" --engram-jobs "$R/j_basalt_eng.json" \
  --engram-tail-dest "${ENGRAM_TAIL_PATH:-}" --engram-head-bytes "${ENGRAM_HEAD_BYTES:-0}" \
  --engram-tail-jobs "$R/j_basalt_eng_tail.json" \
  --engram-size-out "$R/engram_size.txt" > $R/fetch.log 2>&1 || { st "basalt_jobs FAILED"; exit 1; }
ENGRAM_HEAD_SIZE=$(cat $R/engram_size.txt)

# Pre-create 0-block sparse Engram file(s) so PleReader::open sees exact logical sizes during /load
python3 -I -c "
import os
def prep(p, sz):
    if p and int(sz) > 0:
        f = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)
        os.ftruncate(f, int(sz))
        os.close(f)
prep('$ENGRAM_PATH', '$ENGRAM_HEAD_SIZE')
prep('${ENGRAM_TAIL_PATH:-}', '${ENGRAM_TAIL_BYTES:-0}')
"

# Launch parallel build + abliteration fetch + backbone fetch (+ SSD Engram tail if PARALLEL_TAIL=1!)
SSD_WORKERS=$(( WORKERS < 16 ? WORKERS : 16 ))
PID_TAIL=""
if [ "${PARALLEL_TAIL:-0}" = 1 ]; then
  if [ -n "${ENGRAM_TAIL_PATH:-}" ] && [ -s "$R/j_basalt_eng_tail.json" ]; then
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
[[ "$MODEL_PATH" == /content/* ]] && BB_WORKERS=$SSD_WORKERS
{ python3 $V/pfetch.py $R/j_basalt_bb.json --workers $BB_WORKERS --chunk-mb 64 --tag basalt-bb --log $R/fetch.log \
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
  http://127.0.0.1:$PORT/load > $R/load.json 2>&1 && st "weights loaded" || { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "weights load FAILED ($(tail -5 $R/server.log 2>/dev/null | tr '\n' ' '))"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${VRAM:-0}" -lt 60000 ]; then
  [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true
  st "GUARD: VRAM ${VRAM} MiB after /load FAILED"; exit 1
fi
if [ -n "${BASALT_PUNCH_FILE:-}" ]; then
  python3 $V/basalt_jobs.py --backbone-dest "$MODEL_PATH" --punch-sections dspark/gguf dense > $R/punch.txt 2>&1 || true
fi
st "pin: VRAM ${VRAM} MiB, model $(du -sh "$MODEL_PATH" | awk '{print $1}'), mode=$STAGE_MODE"

# Stream RAM Engram (full in ram+ram, head in ram+hybrid/ssd+hybrid) and/or sequential SSD tail
if [[ "$ENGRAM_PATH" == /mnt/ram/* ]]; then
  ( for k in $(seq 1 600); do [ -f $R/READY ] && exit 0; a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
      [ "$a" -lt 1200 ] && { for p in $(pgrep -f "[p]fetch.py"); do kill $p; done; st "GUARD: ${a} MiB RAM available, engram fetch stopped FAILED"; exit 0; }; sleep 2; done ) &
fi

# In ssd+hybrid (where PARALLEL_TAIL=0), start SSD tail & RAM head concurrently now that /content backbone is punched!
if [ "${PARALLEL_TAIL:-0}" = 0 ] && [ -n "${ENGRAM_TAIL_PATH:-}" ] && [ -s "$R/j_basalt_eng_tail.json" ]; then
  { python3 $V/pfetch.py $R/j_basalt_eng_tail.json --workers $SSD_WORKERS --chunk-mb 64 --tag basalt-eng-tail --log $R/fetch_tail.log \
    && st "engram tail done ($ENGRAM_TAIL_PATH)" || st "engram tail FAILED"; } &
  PID_TAIL=$!
fi

# Stream primary Engram file unless it was already started in background (ram+ssd with PARALLEL_TAIL=1)
if [ "$STAGE_MODE" != "ram+ssd" ] || [ "${PARALLEL_TAIL:-0}" = 0 ]; then
  ENG_WORKERS=$WORKERS
  [[ "$ENGRAM_PATH" == /content/* ]] && ENG_WORKERS=$SSD_WORKERS
  python3 $V/pfetch.py $R/j_basalt_eng.json --workers $ENG_WORKERS --chunk-mb 64 --tag basalt-eng --log $R/fetch.log \
    && st "engram done ($ENGRAM_PATH)" || { [ -n "$PID_TAIL" ] && kill $PID_TAIL 2>/dev/null || true; st "engram FAILED"; exit 1; }
fi

if [ -n "$PID_TAIL" ]; then
  wait $PID_TAIL
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
