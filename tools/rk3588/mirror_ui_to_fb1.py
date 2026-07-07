from __future__ import annotations

import mmap
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

DISPLAY: Final = ":0"
FBDEV: Final = Path("/dev/fb1")
FPS: Final = 12.0
RGB565_BYTES: Final = 2
WINDOW_NAME: Final = "UI"


def interrupt(_signum: int, _frame: object) -> None:
  raise KeyboardInterrupt


@dataclass(frozen=True, slots=True)
class Framebuffer:
  width: int
  height: int
  stride: int
  size: int


@dataclass(frozen=True, slots=True)
class Geometry:
  x: int
  y: int
  width: int
  height: int

  @property
  def frame_size(self) -> int:
    return self.width * self.height * RGB565_BYTES

  @property
  def stride(self) -> int:
    return self.width * RGB565_BYTES


@dataclass(frozen=True, slots=True)
class Window:
  xid: str
  width: int
  height: int


def framebuffer() -> Framebuffer:
  root = Path("/sys/class/graphics") / FBDEV.name
  width, height = (int(value) for value in (root / "virtual_size").read_text().strip().split(",", 1))
  bits_per_pixel = int((root / "bits_per_pixel").read_text().strip())
  if bits_per_pixel != 16:
    raise RuntimeError(f"{FBDEV} must be 16bpp, got {bits_per_pixel}")
  stride = width * RGB565_BYTES
  return Framebuffer(width=width, height=height, stride=stride, size=stride * height)


def xauth() -> Path | None:
  paths = sorted(Path("/run/user/1000").glob(".mutter-Xwaylandauth.*"))
  if paths:
    return paths[0]
  path = Path.home() / ".Xauthority"
  return path if path.exists() else None


def xenv(auth: Path | None) -> dict[str, str]:
  env = os.environ.copy()
  env["DISPLAY"] = DISPLAY
  if auth is not None:
    env["XAUTHORITY"] = str(auth)
  return env


def find_window(env: dict[str, str]) -> Window:
  output = subprocess.check_output(["xwininfo", "-root", "-tree"], env=env, text=True)
  escaped = re.escape(WINDOW_NAME)
  patterns = (
    r"^\s*(0x[0-9a-fA-F]+)\s+\"" + escaped + r"\": \(\"" + escaped + r"\" \"" + escaped + r"\"\)\s+(\d+)x(\d+)",
    r"^\s*(0x[0-9a-fA-F]+)\s+\"" + escaped + r"\".*\s+(\d+)x(\d+)",
  )
  for pattern in patterns:
    for line in output.splitlines():
      match = re.search(pattern, line)
      if match is not None:
        return Window(xid=match.group(1), width=int(match.group(2)), height=int(match.group(3)))
  raise RuntimeError(f"X window not found: {WINDOW_NAME!r}")


def geometry(window: Window, fb: Framebuffer) -> Geometry:
  scale = min(fb.width / window.width, fb.height / window.height)
  width = max(1, int(round(window.width * scale)))
  height = max(1, int(round(window.height * scale)))
  return Geometry(x=(fb.width - width) // 2, y=(fb.height - height) // 2, width=width, height=height)


def gst_command(window: Window, geo: Geometry) -> list[str]:
  return [
    "gst-launch-1.0", "-q",
    "ximagesrc", f"xid={window.xid}", "use-damage=0", "show-pointer=false",
    "!", f"video/x-raw,framerate={max(1, int(round(FPS)))}/1",
    "!", "videoscale",
    "!", "videoconvert",
    "!", f"video/x-raw,format=RGB16,width={geo.width},height={geo.height}",
    "!", "fdsink", "fd=1", "sync=false", "async=false",
  ]


def read_frame(stream: BinaryIO, size: int) -> bytes | None:
  chunks: list[bytes] = []
  remaining = size
  while remaining > 0:
    chunk = stream.read(remaining)
    if not chunk:
      return None
    chunks.append(chunk)
    remaining -= len(chunk)
  return b"".join(chunks)


def write_frame(dst: mmap.mmap, fb: Framebuffer, geo: Geometry, frame: bytes) -> None:
  start = geo.y * fb.stride + geo.x * RGB565_BYTES
  if geo.width == fb.width:
    dst[start:start + len(frame)] = frame
    return
  for row in range(geo.height):
    src = row * geo.stride
    dst_row = start + row * fb.stride
    dst[dst_row:dst_row + geo.stride] = frame[src:src + geo.stride]


def stop(proc: subprocess.Popen[bytes]) -> None:
  proc.terminate()
  try:
    proc.wait(timeout=2)
  except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()


def run() -> None:
  signal.signal(signal.SIGTERM, interrupt)
  fb = framebuffer()
  env = xenv(xauth())
  window = find_window(env)
  geo = geometry(window, fb)
  print(
    f"mirror {window.xid} {window.width}x{window.height} -> "
    f"{FBDEV} {fb.width}x{fb.height} content={geo.width}x{geo.height}+{geo.x}+{geo.y} fps={FPS}",
    flush=True,
  )

  proc = subprocess.Popen(gst_command(window, geo), env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
  if proc.stdout is None:
    stop(proc)
    raise RuntimeError("capture stream unavailable")

  frames = 0
  with FBDEV.open("r+b", buffering=0) as fb_file:
    fb_map = mmap.mmap(fb_file.fileno(), fb.size, access=mmap.ACCESS_WRITE)
    fb_map[:] = bytes(fb.size)
    try:
      while True:
        frame = read_frame(proc.stdout, geo.frame_size)
        if frame is None:
          break
        write_frame(fb_map, fb, geo, frame)
        frames += 1
    finally:
      fb_map[:] = bytes(fb.size)
      fb_map.flush()
      stop(proc)
      print(f"frames={frames}", flush=True)


def main() -> int:
  try:
    run()
  except KeyboardInterrupt:
    return 130
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
