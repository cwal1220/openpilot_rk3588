#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Final

BENCH_PROCESSES: Final[tuple[str, ...]] = (
  "webcamerad",
  "ui",
  "soundd",
  "modeld",
  "calibrationd",
  "plannerd",
)
PARAM_KEYS: Final[tuple[str, ...]] = (
  "CarParams",
  "CarParamsCache",
  "CarParamsPersistent",
)
PUBLISH_SERVICES: Final[tuple[str, ...]] = (
  "controlsState",
  "deviceState",
  "pandaStates",
  "carParams",
  "carState",
  "carControl",
  "liveParameters",
  "liveDelay",
  "driverMonitoringState",
  "radarState",
  "selfdriveState",
  "soundPressure",
  "managerState",
)
AFFINITY_PROCESSES: Final[frozenset[str]] = frozenset(("modeld", "webcamerad"))


def parse_cpu_list(raw_value: str) -> set[int]:
  cpus: set[int] = set()
  for part in raw_value.split(","):
    token = part.strip()
    if not token:
      continue
    if "-" in token:
      start, end = (int(value) for value in token.split("-", 1))
      cpus.update(range(start, end + 1))
    else:
      cpus.add(int(token))
  if not cpus:
    raise ValueError(f"empty CPU affinity list: {raw_value!r}")
  return cpus


def apply_rk3588_affinity(process_name: str, pid: int) -> None:
  if os.environ.get("OPENPILOT_RK3588_AFFINITY", "1") != "1" or process_name not in AFFINITY_PROCESSES:
    return
  cpus = parse_cpu_list(os.environ.get("OPENPILOT_RK3588_BIG_CORES", "4-7"))
  try:
    os.sched_setaffinity(pid, cpus)
  except OSError as exc:
    print(f"Unable to set {process_name} affinity to {sorted(cpus)}: {exc}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Run a minimal RK3588 webcam/modeld bench harness.")
  parser.add_argument("--duration", type=float, default=60.0, help="seconds to run; <=0 runs until interrupted")
  parser.add_argument("--processes", default=",".join(BENCH_PROCESSES), help="comma-separated manager processes to start")
  parser.add_argument("--road-cam", default="0", help="ROAD_CAM value for webcamerad, default /dev/video0")
  parser.add_argument("--fourcc", default="MJPG", help="WEBCAM_FOURCC value, default MJPG for RK3588 USB smoke")
  parser.add_argument("--speed", type=float, default=15.0, help="published bench car speed in m/s")
  parser.add_argument("--publish-hz", type=float, default=20.0, help="message publisher frequency")
  return parser.parse_args()


def parse_processes(processes: str) -> tuple[str, ...]:
  requested = tuple(process.strip() for process in processes.split(",") if process.strip())
  unknown = sorted(set(requested) - set(BENCH_PROCESSES))
  if unknown:
    raise ValueError(f"Unsupported bench process: {', '.join(unknown)}")
  return requested


def cleanup_bench_prefix(prefix: str) -> None:
  from openpilot.common.params import Params

  param_link = Params().get_param_path()
  if os.path.islink(param_link):
    shutil.rmtree(os.path.realpath(param_link), ignore_errors=True)
    os.remove(param_link)
  shutil.rmtree(Path("/dev/shm") / f"msgq_{prefix}", ignore_errors=True)


def set_webcam_env(args: argparse.Namespace, prefix: str) -> None:
  os.environ["OPENPILOT_PREFIX"] = prefix
  Path("/dev/shm", f"msgq_{prefix}").mkdir(exist_ok=True)
  os.environ["USE_WEBCAM"] = "1"
  os.environ["ROAD_CAM"] = args.road_cam
  os.environ["WEBCAM_FOURCC"] = args.fourcc
  if "DISPLAY" not in os.environ and os.path.exists("/tmp/.X11-unix/X0"):
    os.environ["DISPLAY"] = ":0"
  runtime_dir = f"/run/user/{os.getuid()}"
  if "XAUTHORITY" not in os.environ:
    for xauth_path in sorted(Path(runtime_dir).glob(".mutter-Xwaylandauth.*")):
      os.environ["XAUTHORITY"] = str(xauth_path)
      break
  if "XDG_RUNTIME_DIR" not in os.environ and os.path.isdir(runtime_dir):
    os.environ["XDG_RUNTIME_DIR"] = runtime_dir


def seed_car_params(params):
  from opendbc.car.car_helpers import get_demo_car_params

  CP = get_demo_car_params()
  CP.notCar = False
  CP.openpilotLongitudinalControl = True
  CP.alphaLongitudinalAvailable = False
  cp_bytes = CP.to_bytes()
  for key in PARAM_KEYS:
    params.put(key, cp_bytes, block=True)
  return CP


def seed_onboarding_params(params):
  from openpilot.common.version import terms_version, training_version

  params.put("HasAcceptedTerms", terms_version, block=True)
  params.put("CompletedTrainingVersion", training_version, block=True)


def build_messages(CP, speed: float):
  from openpilot.cereal import log, messaging
  from opendbc.car.structs import car

  msgs = {service: messaging.new_message(service, valid=True) for service in PUBLISH_SERVICES if service != "pandaStates"}
  msgs["pandaStates"] = messaging.new_message("pandaStates", 1, valid=True)

  msgs["pandaStates"].pandaStates[0].ignitionLine = True
  msgs["pandaStates"].pandaStates[0].pandaType = log.PandaState.PandaType.uno

  msgs["carParams"].carParams = CP

  car_state = msgs["carState"].carState
  car_state.canValid = True
  car_state.vEgo = speed
  car_state.vEgoRaw = speed
  car_state.vEgoCluster = speed
  car_state.vCruise = speed * 3.6
  car_state.vCruiseCluster = speed * 3.6
  car_state.standstill = speed < 0.01
  car_state.cruiseState.available = True
  car_state.cruiseState.enabled = True
  car_state.cruiseState.speed = speed
  car_state.gearShifter = car.CarState.GearShifter.drive

  car_control = msgs["carControl"].carControl
  car_control.enabled = True
  car_control.latActive = True
  car_control.longActive = False
  car_control.orientationNED = [0.0, 0.0, 0.0]
  car_control.angularVelocity = [0.0, 0.0, 0.0]

  msgs["controlsState"].controlsState.curvature = 0.0
  msgs["controlsState"].controlsState.desiredCurvature = 0.0

  live_parameters = msgs["liveParameters"].liveParameters
  live_parameters.valid = True
  live_parameters.sensorValid = True
  live_parameters.posenetValid = True
  live_parameters.steerRatio = CP.steerRatio
  live_parameters.stiffnessFactor = 1.0

  live_delay = msgs["liveDelay"].liveDelay
  live_delay.lateralDelay = CP.steerActuatorDelay + 0.2
  live_delay.status = log.LiveDelayData.Status.estimated

  msgs["selfdriveState"].selfdriveState.state = log.SelfdriveState.OpenpilotState.enabled
  msgs["selfdriveState"].selfdriveState.enabled = True
  msgs["selfdriveState"].selfdriveState.active = True
  msgs["selfdriveState"].selfdriveState.engageable = True
  msgs["selfdriveState"].selfdriveState.alertSound = log.SelfdriveState.AudibleAlert.none
  msgs["selfdriveState"].selfdriveState.personality = log.LongitudinalPersonality.standard

  msgs["soundPressure"].soundPressure.soundPressureWeightedDb = 30.0
  return msgs


def publish_messages(pm, msgs, managed_processes, process_names: tuple[str, ...], thermal_config, device_type: str) -> None:
  from openpilot.cereal import log, messaging

  msgs["deviceState"] = messaging.new_message("deviceState", valid=True)
  device_state = msgs["deviceState"].deviceState
  device_state.started = True
  device_state.deviceType = device_type
  for field, value in thermal_config.get_msg().items():
    setattr(device_state, field, value)
  max_temp = max([device_state.memoryTempC, *device_state.cpuTempC, *device_state.gpuTempC, *device_state.pmicTempC], default=0.0)
  device_state.maxTempC = max_temp
  device_state.thermalStatus = log.DeviceState.ThermalStatus.critical if max_temp > 107.0 else log.DeviceState.ThermalStatus.overheated if max_temp > 96.0 else log.DeviceState.ThermalStatus.ok
  manager_states = [managed_processes[name].get_process_state_msg() for name in process_names]
  msgs["managerState"] = messaging.new_message("managerState", valid=True)
  msgs["managerState"].managerState.processes = manager_states
  for service in PUBLISH_SERVICES:
    msg = msgs[service]
    pm.send(service, msg)
    msg.clear_write_flag()


def run_publisher(args: argparse.Namespace, managed_processes, CP, process_names: tuple[str, ...]) -> bool:
  from openpilot.cereal import messaging
  from openpilot.common.hardware import HARDWARE
  from openpilot.common.params import Params
  from openpilot.common.realtime import Ratekeeper

  params = Params()
  pm = messaging.PubMaster(PUBLISH_SERVICES)
  msgs = build_messages(CP, args.speed)
  thermal_config, device_type = HARDWARE.get_thermal_config(), HARDWARE.get_device_type()
  rk = Ratekeeper(args.publish_hz, print_delay_threshold=None)
  start_time = time.monotonic()
  last_cycle_check = 0.0

  while args.duration <= 0.0 or time.monotonic() - start_time < args.duration:
    now = time.monotonic()
    if now - last_cycle_check >= 0.25:
      last_cycle_check = now
      if params.get_bool("OnroadCycleRequested"):
        params.remove("OnroadCycleRequested")
        return True
    publish_messages(pm, msgs, managed_processes, process_names, thermal_config, device_type)
    rk.keep_time()
  return False


def install_signal_handlers() -> None:
  def handler(signum: int, frame) -> None:
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, handler)
  signal.signal(signal.SIGINT, handler)


def main() -> int:
  from openpilot.common.params import Params

  calibration_params = Params().get("CalibrationParams")
  args = parse_args()
  prefix = os.environ.get("OPENPILOT_PREFIX", f"rk3588_{os.getpid()}")
  if not prefix.startswith("rk3588_") or "/" in prefix:
    raise ValueError("OPENPILOT_PREFIX for this bench must start with rk3588_ and contain no slashes")
  process_names = parse_processes(args.processes)
  set_webcam_env(args, prefix)

  from openpilot.system.manager.process_config import managed_processes

  while True:
    Path("/dev/shm", f"msgq_{prefix}").mkdir(exist_ok=True)
    params = Params()
    if calibration_params is not None:
      params.put("CalibrationParams", calibration_params, block=True)
    CP = seed_car_params(params)
    seed_onboarding_params(params)
    started: list[str] = []
    restart_requested = False
    try:
      for name in process_names:
        managed_processes[name].start()
        started.append(name)
        proc = managed_processes[name].proc
        if proc is not None and proc.pid is not None:
          apply_rk3588_affinity(name, proc.pid)
      install_signal_handlers()
      restart_requested = run_publisher(args, managed_processes, CP, process_names)
    except KeyboardInterrupt:
      pass

    runtime_failures = {
      name: managed_processes[name].proc.exitcode
      for name in started
      if managed_processes[name].proc is not None and managed_processes[name].proc.exitcode not in (None, 0)
    }
    exit_codes: dict[str, int | None] = {}
    for name in reversed(started):
      exit_codes[name] = managed_processes[name].stop()
    cleanup_bench_prefix(prefix)

    shutdown_failures = {name: code for name, code in exit_codes.items() if code not in (None, 0)}
    if runtime_failures:
      print(f"Runtime process failures: {runtime_failures}")
      if shutdown_failures:
        print(f"Shutdown process exits: {shutdown_failures}")
      return 1
    if shutdown_failures:
      print(f"Shutdown process exits: {shutdown_failures}")
    if not restart_requested:
      break
    print("Onroad cycle requested; restarting bench processes")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
