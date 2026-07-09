#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." >/dev/null && pwd)"
MIRROR_SCRIPT="${SCRIPT_DIR}/mirror_ui_to_fb1.py"

OPENCL_WARP_LIB_DEFAULT="${HOME}/.openpilot/lib/libopencl_yuv6_warp.so"
RKNN_MODEL_PATH="${ROOT_DIR}/openpilot/selfdrive/modeld/models/driving_supercombo_rk3588.rknn"

prepare_only=0
print_env=0
force_build=0

for arg in "$@"; do
  case "$arg" in
    --prepare-only) prepare_only=1 ;;
    --print-env) print_env=1 ;;
    --force-build) force_build=1 ;;
    *)
      echo "usage: $0 [--prepare-only] [--print-env] [--force-build]" >&2
      exit 2
      ;;
  esac
done

ensure_prebuilt_marker() {
  # Required because the stock launcher runs a full SCons build without this marker.
  [ -e "${ROOT_DIR}/prebuilt" ] || : > "${ROOT_DIR}/prebuilt"
}

apply_runtime_affinity_once() {
  [ "${OPENPILOT_RK3588_AFFINITY:-1}" = "1" ] || return 0
  command -v taskset >/dev/null || return 0

  for pattern in "openpilot.selfdrive.modeld.modeld" "openpilot.system.camerad.webcam.camerad"; do
    pgrep -f "$pattern" | while read -r pid; do
      taskset -pc "${OPENPILOT_RK3588_BIG_CORES}" "$pid" >/dev/null 2>&1 || true
    done
  done
}

start_affinity_watcher() {
  [ "${OPENPILOT_RK3588_AFFINITY:-1}" = "1" ] || return 0
  local parent_pid="$$"
  (
    while kill -0 "$parent_pid" >/dev/null 2>&1; do
      apply_runtime_affinity_once
      sleep 1
    done
  ) &
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

start_fb1_mirror() {
  [ -e /dev/fb1 ] || return 0
  [ -f "$MIRROR_SCRIPT" ] || return 0
  [ -x "${ROOT_DIR}/.venv/bin/python" ] || return 0

  local parent_pid="$$"
  (
    if wait_for_ui_window; then
      "${ROOT_DIR}/.venv/bin/python" "$MIRROR_SCRIPT" &
      mirror_pid="$!"
      while kill -0 "$parent_pid" >/dev/null 2>&1 && kill -0 "$mirror_pid" >/dev/null 2>&1; do
        sleep 1
      done
      kill "$mirror_pid" >/dev/null 2>&1 || true
      wait "$mirror_pid" >/dev/null 2>&1 || true
    else
      echo "warning: UI X window not found; fb1 mirror was not started" >&2
    fi
  ) &
}

OPENCL_WARP_LIB="${OPENPILOT_OPENCL_WARP_LIB:-$OPENCL_WARP_LIB_DEFAULT}"
OPENCL_WARP_SRC="${ROOT_DIR}/tools/rk3588/opencl_yuv6_warp.cpp"

if [ ! -f "$RKNN_MODEL_PATH" ]; then
  echo "RKNN model not found: $RKNN_MODEL_PATH" >&2
  exit 1
fi

if [ ! -f "$OPENCL_WARP_SRC" ]; then
  echo "OpenCL warp source not found: $OPENCL_WARP_SRC" >&2
  exit 1
fi

if ! command -v g++ >/dev/null; then
  echo "g++ not found; cannot build OpenCL warp library" >&2
  exit 1
fi

if [ ! -f /usr/include/CL/cl.h ]; then
  echo "OpenCL headers not found: /usr/include/CL/cl.h" >&2
  exit 1
fi

mkdir -p "$(dirname "$OPENCL_WARP_LIB")"
if [ "$force_build" -eq 1 ] || [ ! -f "$OPENCL_WARP_LIB" ] || [ "$OPENCL_WARP_SRC" -nt "$OPENCL_WARP_LIB" ]; then
  g++ -std=c++17 -O2 -fPIC -shared -Wall -Wextra -Werror \
    "$OPENCL_WARP_SRC" -lOpenCL -o "$OPENCL_WARP_LIB"
fi

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  "${ROOT_DIR}/.venv/bin/python" - <<'PY'
from rknnlite.api import RKNNLite
import sys
from tinygrad.tensor import Tensor

assert (Tensor([1], device="CL") + 1).numpy()[0] == 2
RKNNLite
print("RK3588 OpenCL/RKNN runtime check OK", file=sys.stderr)
PY
fi

export USE_WEBCAM="${USE_WEBCAM:-1}"
export BIG="${BIG:-1}"
export ROAD_CAM="${ROAD_CAM:-11}"
export WEBCAM_FOURCC="${WEBCAM_FOURCC:-NV12}"
export OPENPILOT_MODELD_RKNN="${OPENPILOT_MODELD_RKNN:-1}"
export OPENPILOT_MODELD_OPENCL_WARP="${OPENPILOT_MODELD_OPENCL_WARP:-1}"
export OPENPILOT_RK3588_AFFINITY="${OPENPILOT_RK3588_AFFINITY:-1}"
export OPENPILOT_RK3588_BIG_CORES="${OPENPILOT_RK3588_BIG_CORES:-4-7}"
export OPENPILOT_OPENCL_WARP_LIB="$OPENCL_WARP_LIB"

if [ "$print_env" -eq 1 ]; then
  for var in USE_WEBCAM BIG ROAD_CAM WEBCAM_FOURCC OPENPILOT_MODELD_RKNN \
             OPENPILOT_MODELD_OPENCL_WARP OPENPILOT_RK3588_AFFINITY \
             OPENPILOT_RK3588_BIG_CORES OPENPILOT_OPENCL_WARP_LIB; do
    printf 'export %s=%q\n' "$var" "${!var}"
  done
fi

if [ "$prepare_only" -eq 1 ]; then
  ensure_prebuilt_marker
  exit 0
elif [ "$print_env" -eq 1 ]; then
  exit 0
fi

ensure_prebuilt_marker

if [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  export VIRTUAL_ENV="${ROOT_DIR}/.venv"
  export PATH="${VIRTUAL_ENV}/bin:${PATH}"
fi

if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
  runtime_dir="/run/user/$(id -u)"
  if [ -d "$runtime_dir" ]; then
    export XDG_RUNTIME_DIR="$runtime_dir"
  fi
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

cd "$ROOT_DIR"
start_affinity_watcher
start_fb1_mirror
exec ./launch_openpilot.sh
