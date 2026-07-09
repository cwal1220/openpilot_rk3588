import os

from openpilot.cereal import log
from openpilot.common.hardware.base import HardwareBase, ThermalConfig, ThermalZone

class Pc(HardwareBase):
  def get_device_type(self):
    return "pc"

  def get_network_type(self):
    # some stuff is gated on wifi, so just assume for now
    return log.DeviceState.NetworkType.wifi

  def get_thermal_config(self):
    zone_types = set()
    try:
      for path in os.scandir("/sys/devices/virtual/thermal"):
        if path.name.startswith("thermal_zone"):
          with open(os.path.join(path.path, "type")) as f:
            zone_types.add(f.read().strip())
    except OSError:
      return super().get_thermal_config()

    if "soc-thermal" not in zone_types:
      return super().get_thermal_config()

    return ThermalConfig(cpu=[ThermalZone("bigcore0-thermal"), ThermalZone("bigcore1-thermal"), ThermalZone("littlecore-thermal")],
                         gpu=[ThermalZone("gpu-thermal")],
                         dsp=ThermalZone("npu-thermal"),
                         memory=ThermalZone("soc-thermal"),
                         pmic=[ThermalZone("center-thermal")])
