"""
Termux-Train Resource Monitor
=============================
Continuous telemetry daemon monitoring thermal and memory boundaries on edge devices.
"""
from __future__ import annotations

import glob
import logging
import os
from typing import Optional

from .hardware import get_system_ram_mb
from ..exceptions import ThermalThrottledError, TrainingOutOfMemoryError

logger = logging.getLogger("termux_train.monitor")


class ResourceMonitor:
    """Monitors hardware safety boundaries during intensive on-device training."""

    def __init__(
        self,
        max_safe_temp_c: float = 45.0,
        min_safe_ram_mb: int = 1024,
    ):
        self.max_safe_temp_c = max_safe_temp_c
        self.min_safe_ram_mb = min_safe_ram_mb

    @staticmethod
    def read_battery_temperature_c() -> Optional[float]:
        """Reads current battery temperature via Android sysfs interface."""
        possible_paths = [
            "/sys/class/power_supply/battery/temp",
            "/sys/class/power_supply/battery/batt_temp",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        val = float(f.read().strip())
                        # Values in sysfs are typically in tenths of a degree Celsius (e.g. 350 for 35.0 C)
                        return val / 10.0 if val > 100.0 else val
                except Exception:
                    pass

        # Check thermal zones
        zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
        for zone in zones:
            try:
                with open(zone, "r", encoding="utf-8") as f:
                    val = float(f.read().strip())
                    celsius = val / 1000.0 if val > 10000.0 else (val / 10.0 if val > 100.0 else val)
                    if 20.0 <= celsius <= 100.0:
                        return celsius
            except Exception:
                continue
        return None

    def check_safety_limits(self, stage: str = "training_step") -> None:
        """Asserts that the system is currently within operational safety parameters."""
        # 1. Thermal Check
        temp_c = self.read_battery_temperature_c()
        if temp_c is not None:
            if temp_c >= self.max_safe_temp_c:
                logger.error("[ResourceMonitor] Thermal threshold exceeded: %.1f°C >= %.1f°C", temp_c, self.max_safe_temp_c)
                raise ThermalThrottledError(current_temp_c=temp_c, max_safe_temp_c=self.max_safe_temp_c)

        # 2. Memory (LMK Prevention) Check
        _, avail_mb = get_system_ram_mb()
        if avail_mb < self.min_safe_ram_mb:
            logger.error("[ResourceMonitor] Memory critical: %dMB available < %dMB threshold", avail_mb, self.min_safe_ram_mb)
            raise TrainingOutOfMemoryError(current_mb=avail_mb, limit_mb=self.min_safe_ram_mb, stage=stage)
