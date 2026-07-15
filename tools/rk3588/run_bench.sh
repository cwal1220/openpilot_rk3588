#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." >/dev/null && pwd)"
ENV_SCRIPT="${SCRIPT_DIR}/run_live.sh"
PROCESSES="webcamerad,ui,soundd,modeld,calibrationd,plannerd"

if [ "$#" -ne 0 ]; then
  echo "usage: $(basename "$0")" >&2
  exit 2
fi

prepare_display_env() {
  if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
    local runtime_dir="/run/user/$(id -u)"
    [ -d "$runtime_dir" ] && export XDG_RUNTIME_DIR="$runtime_dir"
  fi

  if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=":0"
  fi

  if [ -z "${XAUTHORITY:-}" ]; then
    for xauth in "${XDG_RUNTIME_DIR:-}/.mutter-Xwaylandauth."* "${HOME}/.Xauthority"; do
      if [ -s "$xauth" ]; then
        export XAUTHORITY="$xauth"
        break
      fi
    done
  fi
}

stop_child() {
  local pid="${1:-}"
  [ -n "$pid" ] || return 0
  kill "$pid" >/dev/null 2>&1 || return 0
  wait "$pid" >/dev/null 2>&1 || true
}

bench_pid=""

cleanup() {
  stop_child "$bench_pid"
}
trap cleanup EXIT INT TERM

"$ENV_SCRIPT" --prepare-only
eval "$("$ENV_SCRIPT" --print-env)"
prepare_display_env

export PYTHONDONTWRITEBYTECODE=1
export USE_WEBCAM=1
export ROAD_CAM=11
export WEBCAM_FOURCC=NV12

cd "$ROOT_DIR"

./.venv/bin/python tools/rk3588/bench.py \
  --duration 0 \
  --processes "$PROCESSES" \
  --road-cam 11 \
  --fourcc NV12 &
bench_pid="$!"

wait "$bench_pid"
