#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import select
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

import cv2 as cv
import numpy as np

EVENT_FORMAT: Final[str] = "llHHI"
EVENT_SIZE: Final[int] = struct.calcsize(EVENT_FORMAT)
EV_KEY: Final[int] = 0x01
BTN_TOUCH: Final[int] = 0x14A
EVIOCGRAB: Final[int] = 1074021776
TOUCH_NAME_HINTS: Final[tuple[str, ...]] = ("usb2iic", "ctp", "touchscreen", "touch")
WINDOW_NAME: Final[str] = "RK3588 Webcam Intrinsic Calibration"


@dataclass(frozen=True, slots=True)
class BoardSpec:
  cols: int
  rows: int
  square_size: float


@dataclass(frozen=True, slots=True)
class InputDevice:
  path: Path
  name: str


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Standalone OpenCV webcam intrinsic calibration for RK3588.")
  subparsers = parser.add_subparsers(dest="command", required=True)

  collect = subparsers.add_parser("collect", help="show direct webcam preview and tap to save checkerboard frames")
  collect.add_argument("--camera", default="0", help="OpenCV camera index or path")
  collect.add_argument("--out-dir", type=Path, default=Path("~/rk3588_intrinsics").expanduser())
  collect.add_argument("--cols", type=int, default=10, help="checkerboard inner corners across columns")
  collect.add_argument("--rows", type=int, default=7, help="checkerboard inner corners across rows")
  collect.add_argument("--target-count", type=int, default=25)
  collect.add_argument("--width", type=int, default=1280)
  collect.add_argument("--height", type=int, default=720)
  collect.add_argument("--fps", type=int, default=25)
  collect.add_argument("--fourcc", default="MJPG")
  collect.add_argument("--no-flip", action="store_true", help="disable 180 degree flip")
  collect.add_argument("--touch-device", default="auto", help="auto, none, or /dev/input/eventX")
  collect.add_argument("--grab-touch", action="store_true", help="prevent taps from reaching other windows")
  collect.add_argument("--min-interval", type=float, default=0.75)

  solve = subparsers.add_parser("solve", help="solve intrinsics from collected frames")
  solve.add_argument("--image-dir", type=Path, default=Path("~/rk3588_intrinsics/raw").expanduser())
  solve.add_argument("--out", type=Path, default=Path("~/rk3588_intrinsics/intrinsics.json").expanduser())
  solve.add_argument("--cols", type=int, default=10, help="checkerboard inner corners across columns")
  solve.add_argument("--rows", type=int, default=7, help="checkerboard inner corners across rows")
  solve.add_argument("--square-size", type=float, default=1.0, help="checker square size; units only affect extrinsics")
  return parser.parse_args()


def device_from_block(block: dict[str, str]) -> list[InputDevice]:
  name = block.get("N", "").removeprefix('Name="').removesuffix('"')
  handlers = block.get("H", "").removeprefix("Handlers=")
  return [InputDevice(Path("/dev/input") / handler, name) for handler in handlers.split() if handler.startswith("event")]


def find_touch_devices() -> list[InputDevice]:
  devices: list[InputDevice] = []
  block: dict[str, str] = {}
  for line in Path("/proc/bus/input/devices").read_text().splitlines():
    if not line:
      devices.extend(device_from_block(block))
      block = {}
      continue
    prefix, _, value = line.partition(": ")
    block[prefix] = value.strip()
  devices.extend(device_from_block(block))
  return devices


def choose_touch_device(requested: str) -> InputDevice | None:
  if requested == "none":
    return None
  if requested != "auto":
    return InputDevice(Path(requested), requested)
  devices = find_touch_devices()
  for hint in TOUCH_NAME_HINTS:
    for device in devices:
      if hint in device.name.lower():
        return device
  return None


def open_touch_device(device: InputDevice | None, grab_touch: bool) -> BinaryIO | None:
  if device is None:
    return None
  try:
    touch = device.path.open("rb", buffering=0)
  except OSError as exc:
    print(f"touch disabled: cannot open {device.path}: {exc}", file=sys.stderr)
    return None
  if grab_touch:
    try:
      fcntl.ioctl(touch, EVIOCGRAB, 1)
    except OSError as exc:
      print(f"touch grab failed for {device.path}: {exc}", file=sys.stderr)
  print(f"touch capture: {device.path} ({device.name})", file=sys.stderr)
  return touch


def touch_pressed(touch: BinaryIO | None) -> bool:
  if touch is None:
    return False
  readable, _, _ = select.select([touch], [], [], 0.0)
  pressed = False
  while readable:
    data = touch.read(EVENT_SIZE)
    if len(data) != EVENT_SIZE:
      break
    _, _, event_type, code, value = struct.unpack(EVENT_FORMAT, data)
    pressed = pressed or (event_type == EV_KEY and code == BTN_TOUCH and value == 1)
    readable, _, _ = select.select([touch], [], [], 0.0)
  return pressed


def open_camera(camera: str, width: int, height: int, fps: int, fourcc: str) -> cv.VideoCapture:
  camera_id = int(camera) if camera.isdecimal() else camera
  cap = cv.VideoCapture(camera_id)
  if len(fourcc) == 4:
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*fourcc))
  cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
  cap.set(cv.CAP_PROP_FPS, fps)
  if not cap.isOpened():
    raise RuntimeError(f"could not open camera {camera}")
  print(
    f"camera={camera} actual={cap.get(cv.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv.CAP_PROP_FRAME_HEIGHT):.0f} "
    f"fps={cap.get(cv.CAP_PROP_FPS):.1f}",
    flush=True,
  )
  return cap


def find_corners(gray: np.ndarray, board: BoardSpec) -> tuple[bool, np.ndarray | None]:
  pattern_size = (board.cols, board.rows)
  flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
  ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
  if not ok:
    return False, None
  refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                            (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
  return True, refined


def save_capture(frame: np.ndarray, board: BoardSpec, out_dir: Path, index: int) -> bool:
  raw_dir = out_dir / "raw"
  raw_dir.mkdir(parents=True, exist_ok=True)
  name = f"{index:03d}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
  if not cv.imwrite(str(raw_dir / name), frame):
    raise RuntimeError(f"failed to write {raw_dir / name}")
  print(f"saved {name}", flush=True)
  return True


def next_capture_index(out_dir: Path) -> int:
  raw_dir = out_dir / "raw"
  if not raw_dir.exists():
    return 1
  indexes: list[int] = []
  for image_path in raw_dir.glob("*.jpg"):
    prefix = image_path.name.split("_", 1)[0]
    if prefix.isdecimal():
      indexes.append(int(prefix))
  return max(indexes, default=0) + 1


def draw_status(frame: np.ndarray, saved: int, target_count: int, board: BoardSpec) -> np.ndarray:
  display = frame.copy()
  text = f"{board.cols}x{board.rows} inner corners  saved {saved}/{target_count}  tap=save  q=quit"
  cv.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)
  cv.putText(display, text, (16, 28), cv.FONT_HERSHEY_SIMPLEX, 0.72, (80, 255, 120), 2, cv.LINE_AA)
  return display


def collect(args: argparse.Namespace) -> int:
  os.environ.setdefault("DISPLAY", ":0")
  runtime_dir = f"/run/user/{os.getuid()}"
  if Path(runtime_dir).is_dir():
    os.environ.setdefault("XDG_RUNTIME_DIR", runtime_dir)
  board = BoardSpec(args.cols, args.rows, 1.0)
  touch = open_touch_device(choose_touch_device(args.touch_device), args.grab_touch)
  cap = open_camera(args.camera, args.width, args.height, args.fps, args.fourcc)
  cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
  cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

  saved = 0
  next_index = next_capture_index(args.out_dir)
  last_capture = 0.0
  fullscreen_retries = 30
  try:
    while args.target_count <= 0 or saved < args.target_count:
      ok, frame = cap.read()
      if not ok:
        continue
      if not args.no_flip:
        frame = cv.flip(frame, -1)
      cv.imshow(WINDOW_NAME, draw_status(frame, saved, args.target_count, board))
      if fullscreen_retries > 0:
        cv.setWindowProperty(WINDOW_NAME, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        fullscreen_retries -= 1
      key = cv.waitKey(1) & 0xFF
      if key == ord("q"):
        break
      trigger = key in (ord(" "), 13) or touch_pressed(touch)
      if not trigger or time.monotonic() - last_capture < args.min_interval:
        continue
      last_capture = time.monotonic()
      if save_capture(frame, board, args.out_dir, next_index):
        saved += 1
        next_index += 1
  finally:
    cap.release()
    if touch is not None:
      touch.close()
    cv.destroyAllWindows()
  print(f"collected usable frames: {saved}")
  return 0 if saved > 0 else 1


def make_object_points(board: BoardSpec) -> np.ndarray:
  points = np.zeros((board.rows * board.cols, 3), np.float32)
  points[:, :2] = np.mgrid[0:board.cols, 0:board.rows].T.reshape(-1, 2)
  return points * board.square_size


def solve(args: argparse.Namespace) -> int:
  board = BoardSpec(args.cols, args.rows, args.square_size)
  image_paths = sorted(args.image_dir.glob("*.jpg"))
  object_points: list[np.ndarray] = []
  image_points: list[np.ndarray] = []
  image_size: tuple[int, int] | None = None
  template_points = make_object_points(board)
  for image_path in image_paths:
    frame = cv.imread(str(image_path))
    if frame is None:
      continue
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ok, corners = find_corners(gray, board)
    if not ok or corners is None:
      print(f"skip no checkerboard: {image_path.name}", file=sys.stderr)
      continue
    image_size = gray.shape[::-1]
    object_points.append(template_points)
    image_points.append(corners)
  if image_size is None or len(image_points) < 8:
    print(f"need at least 8 checkerboard detections, got {len(image_points)}", file=sys.stderr)
    return 1
  rms, camera_matrix, dist_coeffs, _, _ = cv.calibrateCamera(object_points, image_points, image_size, None, None)
  fx, fy, cx, cy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1]), float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
  result = {
    "image_size": {"width": image_size[0], "height": image_size[1]},
    "checkerboard": {"cols": board.cols, "rows": board.rows, "square_size": board.square_size},
    "detections": len(image_points),
    "rms_reprojection_error": float(rms),
    "camera_matrix": camera_matrix.tolist(),
    "dist_coeffs": dist_coeffs.reshape(-1).tolist(),
    "openpilot_webcam_camera_config": f"{image_size[0]}x{image_size[1]}x{fx:.6f}x{fy:.6f}x{cx:.6f}x{cy:.6f}",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  return 0


def main() -> int:
  args = parse_args()
  if args.command == "collect":
    return collect(args)
  if args.command == "solve":
    return solve(args)
  raise AssertionError(args.command)


if __name__ == "__main__":
  raise SystemExit(main())
