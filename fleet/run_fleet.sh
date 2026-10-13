#!/usr/bin/env bash
# Launch the VisionIST fleet with plain docker run (compose-equivalent).
# All images are local: sipgisr/visionist-<box>:latest
set -u
cd "$(dirname "$0")"

NET=visionist-fleet
docker network create "$NET" 2>/dev/null || true

run() { # run <name> <hostport> <max_mb> <workers> [--gpu]
  local box=$1 hport=$2 mb=$3 wk=$4 gpu=${5:-}
  docker run -d \
    --name "visionist-$box" \
    --restart unless-stopped \
    --network "$NET" \
    -p "${hport}:8061" \
    -e PORT=8061 -e "MAX_MESSAGE_MB=${mb}" -e "MAX_WORKERS=${wk}" \
    ${gpu:+--gpus all} \
    "sipgisr/visionist-${box//_/-}:latest"
}

for c in \
  "clip      9061 64  4 --gpu" \
  "d4rt      9072 256 2 --gpu" \
  "features  9071 64  4" \
  "lang_sam  9064 64  2 --gpu" \
  "lightglue 9069 64  4 --gpu" \
  "mediapipe 9076 64  4" \
  "moge      9067 128 4 --gpu" \
  "open_clip 9073 64  4 --gpu" \
  "opencv    9065 64  4 --gpu" \
  "sbert     9062 16  8 --gpu" \
  "sfm       9074 64  4" \
  "tapnext   9063 256 2 --gpu" \
  "unimatch  9070 128 2 --gpu" \
  "vggt      9066 128 1 --gpu" \
  "yolo      9068 256 4 --gpu" \
; do
  # shellcheck disable=SC2086
  set -- $c; run "$@" || echo "!! $1 failed"
done

# pycv: hardened (matches compose: read_only, tmpfs, cap DROP, limits, no GPU)
docker run -d \
  --name visionist-pycv --restart unless-stopped \
  --network "$NET" -p 9075:8061 \
  -e PORT=8061 -e MAX_MESSAGE_MB=32 -e MAX_WORKERS=2 \
  --read-only --tmpfs /tmp \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --memory 4g --pids-limit 256 \
  sipgisr/visionist-pycv:latest || echo "!! pycv failed"

# webui: after the boxes
docker run -d \
  --name visionist-webui --restart unless-stopped \
  --network "$NET" -p 8080:8000 \
  -e WEBUI_HOST=0.0.0.0 -e WEBUI_PORT=8000 -e WEBUI_DATA_DIR=/data \
  --add-host host.docker.internal:host-gateway \
  -v "$PWD/data:/data" \
  sipgisr/visionist-webui:latest || echo "!! webui failed"

echo
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep visionist-
