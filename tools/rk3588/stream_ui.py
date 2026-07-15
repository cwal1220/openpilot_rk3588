#!/usr/bin/env python3
from __future__ import annotations

import atexit
import math
import os
import signal
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from flask import Flask, Response, request

DISPLAY: Final = ":0"
STREAM_WIDTH, STREAM_HEIGHT, READ_SIZE = 960, 480, 64 * 1024
Action = Literal["down", "move", "up", "cancel"]

HTML: Final = """<!doctype html><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no"><title>openpilot UI</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#000}body{display:grid;place-items:center}img{width:100%;height:100%;object-fit:contain;touch-action:none;user-select:none;-webkit-user-drag:none}</style>
<img id="ui" src="/stream" alt="openpilot UI"><script>
const ui=document.querySelector('#ui');let activePointerId=null,lastMoveAt=0;
function send(action,event){
  const r=ui.getBoundingClientRect(),scale=Math.min(r.width/960,r.height/480),w=960*scale,h=480*scale;
  const left=r.left+(r.width-w)/2,top=r.top+(r.height-h)/2;
  fetch('/input',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action,x:(event.clientX-left)/w,y:(event.clientY-top)/h}),keepalive:true})
}function finish(action,e){if(e.pointerId!==activePointerId)return;send(action,e);ui.releasePointerCapture(e.pointerId);activePointerId=null}
ui.addEventListener('pointerdown',e=>{if(activePointerId!==null)return;activePointerId=e.pointerId;lastMoveAt=e.timeStamp;ui.setPointerCapture(e.pointerId);send('down',e)});ui.addEventListener('pointermove',e=>{if(e.pointerId===activePointerId&&e.timeStamp-lastMoveAt>=50){lastMoveAt=e.timeStamp;send('move',e)}});ui.addEventListener('pointerup',e=>finish('up',e));ui.addEventListener('pointercancel',e=>finish('cancel',e));
</script>"""


@dataclass(frozen=True, slots=True)
class Window:
  xid: str
  width: int
  height: int


@dataclass(frozen=True, slots=True)
class PointerInput:
  action: Action
  x: float
  y: float


def xenv() -> dict[str, str]:
  env = os.environ.copy()
  env["DISPLAY"] = DISPLAY
  auths, fallback = sorted(Path("/run/user/1000").glob(".mutter-Xwaylandauth.*")), Path.home() / ".Xauthority"
  if auths or fallback.exists():
    env["XAUTHORITY"] = str(auths[0] if auths else fallback)
  return env


def find_window() -> Window:
  env = xenv()
  xid = subprocess.check_output(
    ["xdotool", "search", "--onlyvisible", "--class", "^UI$"], env=env, text=True, timeout=1,
  ).splitlines()[0]
  geometry = subprocess.check_output(
    ["xdotool", "getwindowgeometry", "--shell", xid], env=env, text=True, timeout=1,
  )
  values = dict(line.split("=", 1) for line in geometry.splitlines() if "=" in line)
  return Window(xid=xid, width=int(values["WIDTH"]), height=int(values["HEIGHT"]))


def gst_command(window: Window) -> list[str]:
  return [
    "gst-launch-1.0", "-q", "ximagesrc", f"xid={window.xid}", "use-damage=0", "show-pointer=false",
    "!", "video/x-raw,framerate=30/1", "!", "videoscale", "!", "videoconvert",
    "!", f"video/x-raw,format=NV12,width={STREAM_WIDTH},height={STREAM_HEIGHT}",
    "!", "mppjpegenc", "rc-mode=fixqp", "q-factor=75", "zero-copy-pkt=true",
    "!", "multipartmux", "boundary=frame",
    "!", "fdsink", "fd=1", "sync=false", "async=false",
  ]


def parse_input(raw: object) -> PointerInput | None:
  if not isinstance(raw, dict) or set(raw) != {"action", "x", "y"}:
    return None
  action, x, y = raw["action"], raw["x"], raw["y"]
  if action not in ("down", "move", "up", "cancel"):
    return None
  if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
    return None
  if not math.isfinite(x) or not math.isfinite(y):
    return None
  return PointerInput(action=action, x=min(1.0, max(0.0, float(x))), y=min(1.0, max(0.0, float(y))))


def stop_process(process: subprocess.Popen[bytes]) -> None:
  if process.poll() is not None:
    return
  process.terminate()
  try:
    process.wait(timeout=2)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait()


class Runtime:
  def __init__(self) -> None:
    self._window: Window | None = None
    self._button_down = False
    self._pointer_lock = threading.RLock()
    self._input_lock, self._stream_lock, self._process_lock = threading.Lock(), threading.Lock(), threading.Lock()
    self._active_process: subprocess.Popen[bytes] | None = None

  def window(self) -> Window:
    if self._window is None:
      self._window = find_window()
    return self._window

  def input(self, event: PointerInput) -> bool:
    if not self._input_lock.acquire(blocking=event.action != "move"):
      return False
    try:
      with self._pointer_lock:
        window = self.window()
        x, y = round(event.x * (window.width - 1)), round(event.y * (window.height - 1))
        env = xenv()
        subprocess.run(["xdotool", "mousemove", "--window", window.xid, str(x), str(y)], env=env, check=True, timeout=1)
        match event.action:
          case "down":
            subprocess.run(["xdotool", "mousedown", "1"], env=env, check=True, timeout=1)
            self._button_down = True
          case "up" | "cancel":
            self.release()
          case "move":
            pass
      return True
    finally:
      self._input_lock.release()

  def release(self, *, best_effort: bool = False) -> None:
    with self._pointer_lock:
      if not self._button_down:
        return
      try:
        subprocess.run(["xdotool", "mouseup", "1"], env=xenv(), check=True, timeout=1)
      except (OSError, subprocess.CalledProcessError):
        if best_effort:
          return
        raise
      self._button_down = False

  def _stop_stream(self, process: subprocess.Popen[bytes] | None = None) -> None:
    with self._process_lock:
      active = self._active_process
      if active is None or process is not None and active is not process:
        return
      self._active_process = None
    stop_process(active)

  def shutdown(self) -> None:
    self._stop_stream()
    self.release(best_effort=True)

  def start_stream(self) -> Iterator[bytes] | None:
    if not self._stream_lock.acquire(blocking=False):
      return None
    try:
      with self._process_lock:
        process = subprocess.Popen(
          gst_command(self.window()), env=xenv(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if process.stdout is None:
          stop_process(process)
          raise OSError("capture stream unavailable")
        self._active_process = process
    except (OSError, subprocess.SubprocessError, IndexError, KeyError, ValueError):
      self.release(best_effort=True)
      self._stream_lock.release()
      raise

    def chunks() -> Iterator[bytes]:
      try:
        while chunk := os.read(process.stdout.fileno(), READ_SIZE):
          yield chunk
      finally:
        process.stdout.close()
        self.release(best_effort=True)
        self._stop_stream(process)
        self._stream_lock.release()

    return chunks()


def create_app(runtime: Runtime | None = None) -> Flask:
  app = Flask(__name__)
  app.config["MAX_CONTENT_LENGTH"] = 1024
  state = runtime or Runtime()

  @app.get("/")
  def index() -> str:
    return HTML

  @app.get("/stream")
  def stream() -> Response | tuple[str, int]:
    try:
      chunks = state.start_stream()
    except (OSError, subprocess.SubprocessError, IndexError, KeyError, ValueError):
      return "", 503
    if chunks is None:
      return "", 409
    return Response(chunks, content_type="multipart/x-mixed-replace; boundary=frame", direct_passthrough=True)

  @app.post("/input")
  def pointer_input() -> tuple[str, int]:
    event = parse_input(request.get_json(silent=True))
    if event is None:
      return "", 400
    try:
      if not state.input(event):
        return "", 503
    except (OSError, subprocess.SubprocessError, IndexError, KeyError, ValueError):
      return "", 503
    return "", 204

  return app


def install_cleanup(runtime: Runtime) -> None:
  def handle_signal(signum: int, _frame: object) -> None:
    runtime.shutdown()
    raise SystemExit(128 + signum)

  atexit.register(runtime.shutdown)
  signal.signal(signal.SIGTERM, handle_signal)
  signal.signal(signal.SIGINT, handle_signal)


if __name__ == "__main__":
  runtime = Runtime()
  install_cleanup(runtime)
  create_app(runtime).run(host="0.0.0.0", port=8080, threaded=True, debug=False, use_reloader=False)
