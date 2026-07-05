from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

import numpy as np


DEFAULT_OPENCL_WARP_LIB = os.path.expanduser("~/.openpilot/lib/libopencl_yuv6_warp.so")


@dataclass(frozen=True, slots=True)
class OpenClWarpConfig:
  src_w: int
  src_h: int
  dst_w: int
  dst_h: int


class OpenClYuv6Warp:
  def __init__(self, config: OpenClWarpConfig):
    self.config = config
    self.frame_size = config.src_w * config.src_h * 3 // 2
    self.output_shape = (6, config.dst_h // 2, config.dst_w // 2)
    lib_path = os.getenv("OPENPILOT_OPENCL_WARP_LIB", DEFAULT_OPENCL_WARP_LIB)
    if not os.path.isfile(lib_path):
      raise FileNotFoundError(f"OpenCL warp library not found: {lib_path}; run tools/rk3588/run_openpilot_opencl_rknn.sh --prepare-only")
    self.lib = ctypes.CDLL(lib_path)
    self._configure_api()
    self._error = ctypes.create_string_buffer(4096)
    self._handle = self.lib.opencl_yuv6_warp_create(
      config.src_w, config.src_h, config.dst_w, config.dst_h, self._error, len(self._error)
    )
    if not self._handle:
      raise RuntimeError(self._error.value.decode("utf-8", errors="replace"))

  def _configure_api(self) -> None:
    self.lib.opencl_yuv6_warp_create.argtypes = [
      ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t
    ]
    self.lib.opencl_yuv6_warp_create.restype = ctypes.c_void_p
    self.lib.opencl_yuv6_warp_run.argtypes = [
      ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
      ctypes.c_char_p, ctypes.c_size_t,
    ]
    self.lib.opencl_yuv6_warp_run.restype = ctypes.c_int
    self.lib.opencl_yuv6_warp_destroy.argtypes = [ctypes.c_void_p]
    self.lib.opencl_yuv6_warp_destroy.restype = None

  def _run_frame(self, frame: np.ndarray, matrix: np.ndarray, output: np.ndarray) -> None:
    if frame.size != self.frame_size:
      raise ValueError(f"Unexpected NV12 frame size: {frame.size}, expected {self.frame_size}")
    frame_u8 = np.ascontiguousarray(frame, dtype=np.uint8)
    matrix_f32 = np.ascontiguousarray(matrix, dtype=np.float32)
    uv_matrix = matrix_f32 * np.array(
      [[1.0, 1.0, 0.5], [1.0, 1.0, 0.5], [2.0, 2.0, 1.0]], dtype=np.float32
    )
    uv_matrix_f32 = np.ascontiguousarray(uv_matrix, dtype=np.float32)
    output_u8 = np.ascontiguousarray(output, dtype=np.uint8)
    ret = self.lib.opencl_yuv6_warp_run(
      self._handle,
      ctypes.c_void_p(frame_u8.ctypes.data),
      ctypes.c_void_p(matrix_f32.ctypes.data),
      ctypes.c_void_p(uv_matrix_f32.ctypes.data),
      ctypes.c_void_p(output_u8.ctypes.data),
      self._error,
      len(self._error),
    )
    if ret != 0:
      raise RuntimeError(self._error.value.decode("utf-8", errors="replace"))
    if output_u8.ctypes.data != output.ctypes.data:
      output[:] = output_u8

  def warp_pair(self, frames: tuple[np.ndarray, np.ndarray], transforms: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    output = np.empty((2, *self.output_shape), dtype=np.uint8)
    self._run_frame(frames[0], transforms[0], output[0])
    self._run_frame(frames[1], transforms[1], output[1])
    return output

  def __del__(self) -> None:
    handle = getattr(self, "_handle", None)
    if handle:
      self.lib.opencl_yuv6_warp_destroy(handle)
      self._handle = None
