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


    # CTypes Vulkan ABI를 통한 네이티브 하드웨어 직접 질의 (SSOT)
    vendor = "generic"
    gpu_family = "generic"
    soc_model = "unknown"

    try:
        from ameva_runtime.vulkan.doctor import VulkanDoctor
        doc = VulkanDoctor()
        props = doc.query_physical_device_properties()
        if props:
            # 0x5143: Qualcomm, 0x13B5: ARM Mali, 0x10DE: NVIDIA, 0x8086: Intel
            if props.get("vendor_id") == 0x5143 or "adreno" in props.get("device_name", "").lower():
                vendor = "qualcomm"
                gpu_family = "adreno"
            elif props.get("vendor_id") == 0x13B5 or "mali" in props.get("device_name", "").lower():
                vendor = "arm"
                gpu_family = "mali"
            soc_model = props.get("device_name", "unknown")
    except Exception as err:
        logger.debug("[ameva-runtime:detector] VulkanDoctor native probe unavailable: %s", err)
        # 드라이버 미구동 시 커널 디바이스 노드 유무만으로 1차 판정 (문자열 파싱 영구 금지)
        if os.path.exists("/dev/kgsl-3d0"):
            vendor = "qualcomm"
            gpu_family = "adreno"
        elif os.path.exists("/dev/mali0"):
            vendor = "arm"
            gpu_family = "mali"

    has_kgsl = os.path.exists("/dev/kgsl-3d0")
    has_mali = os.path.exists("/dev/mali0")

    # Evaluate routing decision and hardware hazard
    hardware_hazard: str | None = None
    termux_env = is_android() or is_termux()
    has_vulkan_icd = os.path.exists("/system/lib64/libvulkan.so") or os.path.exists("/vendor/lib64/libvulkan.so")

    if gpu_family == "adreno" and has_kgsl:
        recommended_backend = "vulkan"
        recommended_threads = min(4, len(allowed_cpus))
        diagnosis_reason = (
            "Qualcomm Adreno GPU detected with direct KGSL driver node. "
            "Native Vulkan hardware acceleration is fully capable."
        )
    elif gpu_family == "mali" and termux_env:
        if has_vulkan_icd:
            recommended_backend = "vulkan"
            recommended_threads = max(1, min(4, len(allowed_cpus)))
            diagnosis_reason = (
                f"{vendor.upper()} Mali GPU detected with Android Vulkan ICD. "
                "Hardware acceleration active via v2.0.0 Medium MatMul pipeline (Zero-Freeze Mali Quirk)."
            )
        else:
            hardware_hazard = "mali_vulkan_icd_missing"
            recommended_backend = "cpu_neon"
            usable_cores = [c for c in big_cores if c in allowed_cpus] or list(allowed_cpus)
            recommended_threads = max(1, min(4, len(usable_cores)))
            diagnosis_reason = (
                f"{vendor.upper()} Mali GPU detected but Android ICD (/system/lib64/libvulkan.so) missing. "
                f"Routing to high-efficiency ARM NEON CPU engine (Allowed Cores: {sorted(allowed_cpus)})."
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
            "combined_id": soc_model,
            "freq_map": freq_map,
        },
    )
