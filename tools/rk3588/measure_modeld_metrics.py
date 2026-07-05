#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
import time
from typing import Final

from openpilot.cereal import messaging

NS_TO_MS: Final[float] = 1e-6
SEC_TO_MS: Final[float] = 1000.0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Summarize live modelV2 execution metrics.")
  parser.add_argument("--duration", type=float, default=30.0, help="seconds to collect; <=0 runs until --samples is reached")
  parser.add_argument("--samples", type=int, default=0, help="stop after this many modelV2 samples; 0 disables the limit")
  parser.add_argument("--timeout", type=int, default=1000, help="SubMaster poll timeout in milliseconds")
  return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
  if not values:
    return 0.0
  if len(values) == 1:
    return values[0]
  sorted_values = sorted(values)
  rank = (len(sorted_values) - 1) * pct
  lower = int(rank)
  upper = min(lower + 1, len(sorted_values) - 1)
  weight = rank - lower
  return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def format_stats(label: str, values: list[float], unit: str) -> str:
  if not values:
    return f"{label}: no samples"
  return (
    f"{label}: count={len(values)} "
    f"mean={statistics.fmean(values):.3f}{unit} "
    f"p50={percentile(values, 0.50):.3f}{unit} "
    f"p95={percentile(values, 0.95):.3f}{unit} "
    f"max={max(values):.3f}{unit}"
  )


def should_stop(start_time: float, duration: float, samples: int, count: int) -> bool:
  duration_done = duration > 0.0 and time.monotonic() - start_time >= duration
  samples_done = samples > 0 and count >= samples
  return duration_done or samples_done


def collect(args: argparse.Namespace) -> tuple[list[float], list[float], list[float], list[float]]:
  sm = messaging.SubMaster(["modelV2"], poll="modelV2")
  execution_ms: list[float] = []
  frame_drop: list[float] = []
  publish_period_ms: list[float] = []
  wall_period_ms: list[float] = []
  last_log_mono_time = 0
  last_wall_time = 0.0
  start_time = time.monotonic()

  while not should_stop(start_time, args.duration, args.samples, len(execution_ms)):
    sm.update(args.timeout)
    if not sm.updated["modelV2"]:
      continue

    now = time.monotonic()
    log_mono_time = sm.logMonoTime["modelV2"]
    model = sm["modelV2"]
    execution_ms.append(model.modelExecutionTime * SEC_TO_MS)
    frame_drop.append(model.frameDropPerc)

    if last_log_mono_time > 0 and log_mono_time > last_log_mono_time:
      publish_period_ms.append((log_mono_time - last_log_mono_time) * NS_TO_MS)
    if last_wall_time > 0.0:
      wall_period_ms.append((now - last_wall_time) * SEC_TO_MS)

    last_log_mono_time = log_mono_time
    last_wall_time = now

  return execution_ms, frame_drop, publish_period_ms, wall_period_ms


def main() -> int:
  args = parse_args()
  execution_ms, frame_drop, publish_period_ms, wall_period_ms = collect(args)
  if not execution_ms:
    print("No modelV2 samples received.")
    return 1

  print(format_stats("model_execution", execution_ms, "ms"))
  print(format_stats("frame_drop_field", frame_drop, ""))
  print(format_stats("publish_period", publish_period_ms, "ms"))
  print(format_stats("wall_period", wall_period_ms, "ms"))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
