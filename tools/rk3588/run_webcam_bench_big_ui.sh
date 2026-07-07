#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." >/dev/null && pwd)"
ENV_SCRIPT="${SCRIPT_DIR}/run_live_openpilot_rk3588.sh"
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

wait_for_ui_window() {
  local deadline=$((SECONDS + 30))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if xwininfo -root -tree 2>/dev/null | grep -q '"UI": ("UI" "UI")'; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_child() {
  local pid="${1:-}"
  [ -n "$pid" ] || return 0
  kill "$pid" >/dev/null 2>&1 || return 0
  wait "$pid" >/dev/null 2>&1 || true
}

bench_pid=""
mirror_pid=""

cleanup() {
  stop_child "$mirror_pid"
  stop_child "$bench_pid"
}
trap cleanup EXIT INT TERM

"$ENV_SCRIPT" --prepare-only
eval "$("$ENV_SCRIPT" --print-env)"
prepare_display_env

export PYTHONDONTWRITEBYTECODE=1
export BIG=1
export USE_WEBCAM=1
export ROAD_CAM=0
export WEBCAM_FOURCC=MJPG

cd "$ROOT_DIR"

./.venv/bin/python tools/rk3588/run_webcam_bench.py \
  --duration 0 \
  --processes "$PROCESSES" \
  --road-cam 0 \
  --fourcc MJPG &
bench_pid="$!"

if wait_for_ui_window; then
  ./.venv/bin/python tools/rk3588/mirror_ui_to_fb1.py &
  mirror_pid="$!"
else
  echo "warning: UI X window not found; fb1 mirror was not started" >&2
fi

wait "$bench_pid"
