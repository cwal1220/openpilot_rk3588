from __future__ import annotations

import ctypes as C
import os

import numpy as np

RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256
RKNN_SUCC = 0
RKNN_QUERY_IN_OUT_NUM = 0
RKNN_QUERY_INPUT_ATTR = 1
RKNN_QUERY_OUTPUT_ATTR = 2
RKNN_NPU_CORE_0_1_2 = 7
RKNN_INPUT_NAMES = ("img", "big_img", "features_buffer", "desire_pulse", "traffic_convention")


class RknnInputOutputNum(C.Structure):
  _fields_ = [
    ("n_input", C.c_uint32),
    ("n_output", C.c_uint32),
  ]


class RknnTensorAttr(C.Structure):
  _fields_ = [
    ("index", C.c_uint32),
    ("n_dims", C.c_uint32),
    ("dims", C.c_uint32 * RKNN_MAX_DIMS),
    ("name", C.c_char * RKNN_MAX_NAME_LEN),
    ("n_elems", C.c_uint32),
    ("size", C.c_uint32),
    ("fmt", C.c_int),
    ("type", C.c_int),
    ("qnt_type", C.c_int),
    ("fl", C.c_int8),
    ("zp", C.c_int32),
    ("scale", C.c_float),
    ("w_stride", C.c_uint32),
    ("size_with_stride", C.c_uint32),
    ("pass_through", C.c_uint8),
    ("h_stride", C.c_uint32),
  ]


class RknnInput(C.Structure):
  _fields_ = [
    ("index", C.c_uint32),
    ("buf", C.c_void_p),
    ("size", C.c_uint32),
    ("pass_through", C.c_uint8),
    ("type", C.c_int),
    ("fmt", C.c_int),
  ]


class RknnOutput(C.Structure):
  _fields_ = [
    ("want_float", C.c_uint8),
    ("is_prealloc", C.c_uint8),
    ("index", C.c_uint32),
    ("buf", C.c_void_p),
    ("size", C.c_uint32),
  ]


def _attr_name(attr: RknnTensorAttr) -> str:
  return bytes(attr.name).split(b"\0", 1)[0].decode()


def _attr_shape(attr: RknnTensorAttr) -> tuple[int, ...]:
  return tuple(int(attr.dims[i]) for i in range(attr.n_dims))


def make_rknn_input_queues(input_shapes: dict[str, tuple[int, ...]], frame_skip: int) -> dict[str, np.ndarray]:
  img = input_shapes['img']
  n_frames = img[1] // 6
  img_buf_shape = (frame_skip * (n_frames - 1) + 1, 6, img[2], img[3])
  fb = input_shapes['features_buffer']
  dp = input_shapes['desire_pulse']
  return {
    'img_q': np.zeros(img_buf_shape, dtype=np.float16),
    'big_img_q': np.zeros(img_buf_shape, dtype=np.float16),
    'feat_q': np.zeros((frame_skip * fb[1], fb[0], fb[2]), dtype=np.float16),
    'desire_q': np.zeros((frame_skip * dp[1], dp[0], dp[2]), dtype=np.float16),
  }


def make_rknn_inputs(input_shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
  img = input_shapes['img']
  return {
    'img': np.empty((1, int(np.prod(img[1:]))), dtype=np.float16),
    'big_img': np.empty((1, int(np.prod(img[1:]))), dtype=np.float16),
    'features_buffer': np.empty(input_shapes['features_buffer'], dtype=np.float16),
    'desire_pulse': np.empty(input_shapes['desire_pulse'], dtype=np.float16),
    'traffic_convention': np.empty(input_shapes['traffic_convention'], dtype=np.float16),
  }


def make_rknn_runtime_state(input_shapes: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
  return {
    'tfm': np.zeros((3, 3), dtype=np.float32),
    'big_tfm': np.zeros((3, 3), dtype=np.float32),
    'desire': np.zeros((input_shapes['desire_pulse'][2],), dtype=np.float32),
    'traffic_convention': np.zeros(input_shapes['traffic_convention'], dtype=np.float32),
    'action_t': np.zeros(input_shapes['action_t'], dtype=np.float32),
    'prev_feat': np.zeros((input_shapes['features_buffer'][0], input_shapes['features_buffer'][2]), dtype=np.float32),
  }


def shift_queue(buf: np.ndarray, new_val: np.ndarray) -> None:
  count = new_val.shape[0]
  buf[:-count] = buf[count:]
  buf[-count:] = new_val


def copy_sample_skip_np(buf: np.ndarray, frame_skip: int, output: np.ndarray) -> np.ndarray:
  sampled = buf[::frame_skip]
  np.copyto(output.reshape(sampled.shape), sampled, casting="unsafe")
  return output


def copy_desire_np(buf: np.ndarray, frame_skip: int, output: np.ndarray) -> np.ndarray:
  grouped = buf.reshape((-1, frame_skip, *buf.shape[1:]))
  np.max(grouped, axis=1, out=output.reshape(grouped.shape[0], *buf.shape[1:]))
  return output


class RknnPolicy:
  def __init__(self, model_path: str, core_mask: int = RKNN_NPU_CORE_0_1_2):
    if not os.path.isfile(model_path):
      raise FileNotFoundError(f"RKNN model not found: {model_path}")

    self._lib = C.CDLL("/usr/lib/librknnrt.so")
    self._configure_api()
    self._ctx = C.c_uint64(0)
    model_path_bytes = os.fsencode(model_path)
    model_path_c = C.c_char_p(model_path_bytes)
    self._check(self._lib.rknn_init(C.byref(self._ctx), C.cast(model_path_c, C.c_void_p), 0, 0, None), "rknn_init")
    self._check(self._lib.rknn_set_core_mask(self._ctx.value, core_mask), "rknn_set_core_mask")
    self.input_attrs, self.output_attrs = self._query_io_attrs()
    self.input_shapes = {_attr_name(attr): _attr_shape(attr) for attr in self.input_attrs}
    missing_inputs = [name for name in RKNN_INPUT_NAMES if name not in self.input_shapes]
    if missing_inputs:
      raise RuntimeError(f"RKNN metadata missing inputs: {missing_inputs}")
    extra_inputs = set(self.input_shapes) - set(RKNN_INPUT_NAMES)
    if extra_inputs:
      raise RuntimeError(f"Unexpected RKNN metadata inputs: {sorted(extra_inputs)}")
    self._output_buffers = [np.empty((attr.n_elems,), dtype=np.float16) for attr in self.output_attrs]

  def _configure_api(self) -> None:
    self._lib.rknn_init.argtypes = [C.POINTER(C.c_uint64), C.c_void_p, C.c_uint32, C.c_uint32, C.c_void_p]
    self._lib.rknn_init.restype = C.c_int
    self._lib.rknn_set_core_mask.argtypes = [C.c_uint64, C.c_int]
    self._lib.rknn_set_core_mask.restype = C.c_int
    self._lib.rknn_query.argtypes = [C.c_uint64, C.c_int, C.c_void_p, C.c_uint32]
    self._lib.rknn_query.restype = C.c_int
    self._lib.rknn_inputs_set.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnInput)]
    self._lib.rknn_inputs_set.restype = C.c_int
    self._lib.rknn_run.argtypes = [C.c_uint64, C.c_void_p]
    self._lib.rknn_run.restype = C.c_int
    self._lib.rknn_outputs_get.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnOutput), C.c_void_p]
    self._lib.rknn_outputs_get.restype = C.c_int
    self._lib.rknn_outputs_release.argtypes = [C.c_uint64, C.c_uint32, C.POINTER(RknnOutput)]
    self._lib.rknn_outputs_release.restype = C.c_int
    self._lib.rknn_destroy.argtypes = [C.c_uint64]
    self._lib.rknn_destroy.restype = C.c_int

  def _check(self, ret: int, label: str) -> None:
    if ret != RKNN_SUCC:
      raise RuntimeError(f"{label} failed: {ret}")

  def _query_io_attrs(self) -> tuple[list[RknnTensorAttr], list[RknnTensorAttr]]:
    num = RknnInputOutputNum()
    self._check(self._lib.rknn_query(self._ctx.value, RKNN_QUERY_IN_OUT_NUM, C.byref(num), C.sizeof(num)), "query io count")
    return (
      [self._query_attr(RKNN_QUERY_INPUT_ATTR, i) for i in range(num.n_input)],
      [self._query_attr(RKNN_QUERY_OUTPUT_ATTR, i) for i in range(num.n_output)],
    )

  def _query_attr(self, query: int, index: int) -> RknnTensorAttr:
    attr = RknnTensorAttr()
    attr.index = index
    self._check(self._lib.rknn_query(self._ctx.value, query, C.byref(attr), C.sizeof(attr)), "query tensor attr")
    return attr

  def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
    c_inputs = (RknnInput * len(self.input_attrs))()
    keepalive: list[np.ndarray] = []
    for i, attr in enumerate(self.input_attrs):
      name = _attr_name(attr)
      value = np.ascontiguousarray(inputs[name])
      if value.nbytes != attr.size:
        raise RuntimeError(f"RKNN input {name} size mismatch: got {value.nbytes}, expected {attr.size}")
      keepalive.append(value)
      c_inputs[i].index = attr.index
      c_inputs[i].buf = C.c_void_p(value.ctypes.data)
      c_inputs[i].size = value.nbytes
      c_inputs[i].pass_through = 1
      c_inputs[i].type = attr.type
      c_inputs[i].fmt = attr.fmt

    self._check(self._lib.rknn_inputs_set(self._ctx.value, len(c_inputs), c_inputs), "rknn_inputs_set")
    self._check(self._lib.rknn_run(self._ctx.value, None), "rknn_run")
    outputs = (RknnOutput * len(self.output_attrs))()
    for i, attr in enumerate(self.output_attrs):
      outputs[i].want_float = 0
      outputs[i].is_prealloc = 1
      outputs[i].index = attr.index
      outputs[i].buf = C.c_void_p(self._output_buffers[i].ctypes.data)
      outputs[i].size = self._output_buffers[i].nbytes
    self._check(self._lib.rknn_outputs_get(self._ctx.value, len(outputs), outputs, None), "rknn_outputs_get")
    try:
      return self._output_buffers[0].astype(np.float32).reshape(-1)
    finally:
      self._lib.rknn_outputs_release(self._ctx.value, len(outputs), outputs)

  def release(self) -> None:
    ctx = getattr(self, "_ctx", C.c_uint64(0))
    if ctx.value:
      self._lib.rknn_destroy(self._ctx.value)
      self._ctx = C.c_uint64(0)

  def __del__(self) -> None:
    self.release()
