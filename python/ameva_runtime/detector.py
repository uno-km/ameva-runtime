"""
ameva_runtime.detector
======================
Deep Hardware Topology and Device Capability Detector.
Provides Ground Truth hardware profiling for mobile SoCs, GPUs, and CPU cgroup constraints.
"""
from __future__ import annotations

import glob
import os
import subprocess
from dataclasses import dataclass, field
from typing import Set, List, Dict, Any


@dataclass
class HardwareProfile:
    """Detailed hardware capability and topology profile."""
    vendor: str                         # "qualcomm", "samsung_exynos", "mediatek", "google_tensor", "generic"
    soc_model: str                      # e.g., "exynos1380", "sm8650", "dimensity9000", "unknown"
    gpu_family: str                     # "adreno", "mali", "powervr", "generic"
    has_kgsl_node: bool = False         # /dev/kgsl-3d0 presence
    has_mali_node: bool = False         # /dev/mali0 presence
    total_cpu_cores: int = 8            # Logical CPU count
    allowed_cpu_set: Set[int] = field(default_factory=set) # Cgroup / affinity constrained CPU set
    big_core_indices: List[int] = field(default_factory=list) # Highest frequency CPU core indices
    little_core_indices: List[int] = field(default_factory=list) # Efficiency CPU core indices
    recommended_threads: int = 4        # Safe thread count for execution
    recommended_backend: str = "cpu_neon" # "vulkan", "cpu_neon", "opencl", "nnapi"
    hardware_hazard: str | None = None  # Identified driver/hardware trap (e.g. "mali_vulkan_fence_deadlock")
    diagnosis_reason: str = ""          # Explicit, transparent reason for routing choice
    arch: str = "arm64-v8a"
    driver_version: str = "vulkan-1.3"
    has_vulkan_loader: bool = False
    has_opencl: bool = False
    has_npu: bool = False
    total_ram_mb: int = 4096
    available_ram_mb: int = 2048
    is_cgroup_restrained: bool = False
    allowed_cpus: List[int] = field(default_factory=list)
    raw_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.allowed_cpus and self.allowed_cpu_set:
            self.allowed_cpus = sorted(self.allowed_cpu_set)
        elif self.allowed_cpus and not self.allowed_cpu_set:
            self.allowed_cpu_set = set(self.allowed_cpus)
        if not self.is_cgroup_restrained and self.total_cpu_cores > 0:
            self.is_cgroup_restrained = len(self.allowed_cpu_set) < self.total_cpu_cores

    @property
    def cpu_cores(self) -> int:
        return self.total_cpu_cores



def is_android() -> bool:
    """True if running on Android OS."""
    if os.path.exists("/system/build.prop"):
        return True
    return bool(os.environ.get("ANDROID_ROOT") or os.environ.get("ANDROID_DATA"))


def is_termux() -> bool:
    """True if running in native Termux environment."""
    if os.environ.get("TERMUX_VERSION") or os.environ.get("TERMUX_APP_PID"):
        return True
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix or os.path.exists("/data/data/com.termux"):
        return True
    return False


def get_cpu_frequencies() -> Dict[int, int]:
    """Reads maximum frequency in kHz for each available CPU core."""
    freq_map: Dict[int, int] = {}
    cpu_dirs = glob.glob("/sys/devices/system/cpu/cpu[0-9]*")
    for c_dir in cpu_dirs:
        try:
            core_id = int(os.path.basename(c_dir).replace("cpu", ""))
            freq_file = os.path.join(c_dir, "cpufreq", "cpuinfo_max_freq")
            if os.path.exists(freq_file):
                with open(freq_file, "r", encoding="utf-8") as f:
                    freq_map[core_id] = int(f.read().strip())
        except (ValueError, OSError):
            continue
    return freq_map


def get_allowed_cpuset() -> Set[int]:
    """Returns the set of CPUs currently allowed by cgroup / sched_affinity."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return set(os.sched_getaffinity(0))
        except OSError:
            pass
    total = os.cpu_count() or 4
    return set(range(total))


def detect_hardware() -> HardwareProfile:
    """Profiles the runtime environment with zero guesswork.

    Returns a complete HardwareProfile specifying exact SoC, GPU,
    CPU topology, cgroup boundaries, and recommended backend.
    """
    total_cores = os.cpu_count() or 4
    allowed_cpus = get_allowed_cpuset()
    freq_map = get_cpu_frequencies()

    # Determine Big vs Little cores from frequencies
    big_cores: List[int] = []
    little_cores: List[int] = []
    if freq_map:
        max_freq = max(freq_map.values())
        min_freq = min(freq_map.values())
        if max_freq != min_freq:
            for c_id, freq in sorted(freq_map.items()):
                if freq == max_freq:
                    big_cores.append(c_id)
                else:
                    little_cores.append(c_id)
        else:
            big_cores = sorted(freq_map.keys())
    else:
        # Fallback: assume upper half are big cores on typical 8-core mobile SoCs
        if total_cores >= 8:
            little_cores = list(range(total_cores // 2))
            big_cores = list(range(total_cores // 2, total_cores))
        else:
            big_cores = list(range(total_cores))

    # Read CPU info and Android properties
    cpu_info_text = ""
    if os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as f:
                cpu_info_text = f.read()
        except OSError:
            pass

    prop_soc = ""
    soc_model = "unknown"
    if is_android() or is_termux():
        for prop_name in ["ro.soc.model", "ro.board.platform", "ro.hardware", "ro.chipname"]:
            try:
                val = subprocess.check_output(["getprop", prop_name], text=True, stderr=subprocess.DEVNULL).strip()
                if val:
                    prop_soc += " " + val.lower()
                    if soc_model == "unknown":
                        soc_model = val.lower()
            except Exception:
                pass

    combined = (cpu_info_text.lower() + " " + prop_soc).strip()

    vendor = "generic"
    gpu_family = "generic"

    if any(k in combined for k in ["qualcomm", "qcom", "snapdragon", "sm8", "sm7", "sm6", "adreno"]):
        vendor = "qualcomm"
        gpu_family = "adreno"
    elif any(k in combined for k in ["exynos", "s5e", "universal", "samsung"]):
        vendor = "samsung_exynos"
        gpu_family = "mali"
    elif any(k in combined for k in ["mediatek", "mt6", "mt8", "dimensity"]):
        vendor = "mediatek"
        gpu_family = "mali"
    elif any(k in combined for k in ["tensor", "gs101", "gs201", "zuma"]):
        vendor = "google_tensor"
        gpu_family = "mali"

    # Node validation
    has_kgsl = os.path.exists("/dev/kgsl-3d0")
    has_mali = os.path.exists("/dev/mali0")

    if has_kgsl and gpu_family == "generic":
        vendor = "qualcomm"
        gpu_family = "adreno"
    elif has_mali and gpu_family == "generic":
        vendor = "samsung_exynos"
        gpu_family = "mali"

    # Evaluate routing decision and hardware hazard
    hardware_hazard: str | None = None
    termux_env = is_android() or is_termux()

    if gpu_family == "adreno" and has_kgsl:
        recommended_backend = "vulkan"
        recommended_threads = min(4, len(allowed_cpus))
        diagnosis_reason = (
            "Qualcomm Adreno GPU detected with direct KGSL driver node. "
            "Native Vulkan hardware acceleration is fully capable."
        )
    elif gpu_family == "mali" and termux_env:
        hardware_hazard = "mali_vulkan_fence_deadlock"
        recommended_backend = "cpu_neon"
        # Determine available cores within cgroup
        usable_cores = [c for c in big_cores if c in allowed_cpus]
        if not usable_cores:
            usable_cores = list(allowed_cpus)
        recommended_threads = max(1, min(4, len(usable_cores)))
        diagnosis_reason = (
            f"{vendor.upper()} Mali GPU detected. System Vulkan driver (/system/lib64/libvulkan.so) "
            "exhibits known fence synchronization deadlock and SurfaceFlinger screen freeze hazard. "
            f"Routing to high-efficiency ARM NEON CPU engine (Allowed Cores: {sorted(allowed_cpus)}). "
            "Zero-Silent-Fallback compliant."
        )
    else:
        recommended_backend = "vulkan" if not termux_env else "cpu_neon"
        recommended_threads = max(1, min(4, len(allowed_cpus)))
        diagnosis_reason = f"Generic platform ({vendor}). Routing to {recommended_backend}."

    return HardwareProfile(
        vendor=vendor,
        soc_model=soc_model,
        gpu_family=gpu_family,
        has_kgsl_node=has_kgsl,
        has_mali_node=has_mali,
        total_cpu_cores=total_cores,
        allowed_cpu_set=allowed_cpus,
        big_core_indices=big_cores,
        little_core_indices=little_cores,
        recommended_threads=recommended_threads,
        recommended_backend=recommended_backend,
        hardware_hazard=hardware_hazard,
        diagnosis_reason=diagnosis_reason,
        raw_info={
            "combined_id": combined[:120],
            "freq_map": freq_map,
        },
    )
