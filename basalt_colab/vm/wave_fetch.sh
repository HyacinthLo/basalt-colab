#!/bin/bash
# Stage 1 Micro-Wave hook invoked by Loader::run() (BASALT_WAVE_FETCH_CMD):
# Streams only the exact clipped `experts` spans for the current VRAM wave or ~20 GiB RAM micro-wave
# directly into sparse /mnt/ram/model.basalt via zero-GIL pfetch.py (~3.45 GB/s).
set -euo pipefail
SPANS_JSON="${1:?usage: wave_fetch.sh SPANS_JSON TAG}"
TAG="${2:-wave}"
R=${R:-/content/res}
V=${V:-/content/vm}
: "${MODEL_URL:?}"
: "${MODEL_PATH:?}"
: "${MODEL_FILE_BYTES:?}"
JOBS_JSON="$R/j_wave_${TAG}.json"

BYTES=$(python3 - "$SPANS_JSON" "$JOBS_JSON" "$MODEL_URL" "$MODEL_PATH" "$MODEL_FILE_BYTES" <<'EOF'
import json, sys
spans = json.load(open(sys.argv[1]))
out_path, url, dest, fbytes = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
jobs = []
total = 0
for pair in spans:
    a, b = int(pair[0]), int(pair[1])
    if b > a:
        jobs.append({
            "url": url,
            "dest": dest,
            "dest_size": fbytes,
            "src_off": a,
            "dest_off": a,
            "size": b - a,
        })
        total += b - a
with open(out_path, "w") as f:
    json.dump(jobs, f)
print(total)
EOF
)

[ "${BYTES:-0}" -gt 0 ] || exit 0
python3 "$V/pfetch.py" "$JOBS_JSON" --keepalive --workers "${WORKERS:-64}" --chunk-mb 128 \
  --tag "basalt-${TAG}" --log "$R/fetch_wave.log"
rm -f "$JOBS_JSON"
GIB_STR=$(python3 -c "print(f'{int($BYTES)/1073741824:.2f}')")
ELAPSED=$(( $(date +%s) - ${BOOT_T0:-$(date +%s)} ))
[ "$ELAPSED" -lt 0 ] && ELAPSED=0
echo "${ELAPSED}s $(date +%T) wave fetch ${TAG} done (${GIB_STR} GiB)" >> "$R/timeline.txt"
