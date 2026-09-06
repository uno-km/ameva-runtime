"""
Base Utilities & Common Logic for Ameva Modality Adapters
"""
from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from ..doctor import DiagnosticReport
except ImportError:
    DiagnosticReport = Any

from ameva_runtime.protocol import BindingResult

logger = logging.getLogger("ameva_vulkan_runtime.adapters")

_ADRENO_VENDOR_ID = 0x5143
_MALI_VENDOR_ID = 0x13B5


def resolve_diagnostic_report(report: Any = None, profile: Any = None) -> DiagnosticReport:
    """Resolves DiagnosticReport from either an existing report, a HardwareProfile, or Doctor auto-probe."""
    if report is not None:
        return report
    if profile is not None:
        is_vk = (getattr(profile, "recommended_backend", "") == "vulkan" and not getattr(profile, "hardware_hazard", None))
        gpu_family = getattr(profile, "gpu_family", "generic")
        return DiagnosticReport(
            device_name=getattr(profile, "soc_model", None) or gpu_family,
            vendor_id=_MALI_VENDOR_ID if gpu_family == "mali" else _ADRENO_VENDOR_ID,
            overall_success=is_vk,
            recommended_backend=getattr(profile, "recommended_backend", "cpu_neon"),
            passed_stages=12 if is_vk else 0,
            total_stages=12,
            loader_path="/system/lib64/libvulkan.so",
            hazard=getattr(profile, "hardware_hazard", None),
            allowed_cpus=sorted(list(getattr(profile, "allowed_cpu_set", []))),
            diagnosis_reason=getattr(profile, "diagnosis_reason", ""),
        )
    try:
        from ..doctor import Doctor
        return Doctor().run_diagnostics()
    except Exception as err:
        logger.debug("[ameva-runtime:adapters] Doctor probe fallback to minimal report: %s", err)
        return DiagnosticReport(
            device_name="Unknown",
            vendor_id=0,
            overall_success=False,
            recommended_backend="cpu_neon",
            passed_stages=0,
            total_stages=12,
        )


class BaseAdapter:
    """Base class for all ameva modality adapters providing common diagnostic helpers."""

    @staticmethod
    def resolve_diagnostic_report(report: Any = None, profile: Any = None) -> DiagnosticReport:
        return resolve_diagnostic_report(report, profile)


def _is_vulkan_report(report: Optional[DiagnosticReport]) -> bool:
    if report is None:
        return False
    if not report.device_name or report.device_name in ("Unknown", "None", ""):
        return False
    return bool(
        report.overall_success
        or report.recommended_backend in ("vulkan", "vulkan_driver_only")
        or report.passed_stages >= 7
    )


def check_vulkan_availability_or_raise(
    module: str,
    report: Any,
    is_vk: bool,
    requested_backend: Optional[str] = None,
) -> None:
    """Enforces Fail-Fast: If Vulkan was explicitly requested but is unavailable, raise PlatformNotSupportedError."""
    if requested_backend == "vulkan" and not is_vk:
        device = getattr(report, "device_name", None) or "Unknown"
        diag_reason = getattr(report, "diagnosis_reason", None) or "Vulkan ICD / driver missing or self-test validation failed"
        from ..exceptions import PlatformNotSupportedError
        raise PlatformNotSupportedError(
            f"[{module}] Vulkan acceleration backend explicitly requested, but no valid Vulkan driver/environment is available.\n"
            f"Target Device: {device}\n"
            f"Failure Cause: {diag_reason}\n"
            f"Remediation: Verify Vulkan driver installation or explicitly specify '--device cpu'."
        )


def _make_cpu_binding(
    module: str,
    report: DiagnosticReport,
    config: dict,
    reason: str = "",
) -> BindingResult:
    """Creates a standardized CPU NEON BindingResult when CPU backend is explicitly requested or routed."""
    device = getattr(report, "device_name", None) or "Generic CPU"
    msg = f"[ameva-runtime:{module}] Binding CPU NEON backend (Device: {device}"
    if reason:
        msg += f", Reason: {reason}"
    msg += ")"
    logger.info(msg)
    config["backend"] = "cpu_neon"
    return BindingResult(
        module=module,
        backend="cpu_neon",
        is_vulkan=False,
        device_name=getattr(report, "device_name", "CPU"),
        vendor_id=getattr(report, "vendor_id", 0),
        config=config,
        status="BOUND_CPU_NEON",
    )


# Backward compatibility alias
_make_cpu_fallback = _make_cpu_binding


def _get_optimal_threads() -> int:
    """Returns optimal threads count for big/performance cores."""
    import os
    cpu_count = os.cpu_count() or 8
    return max(1, min(4, cpu_count // 2 if cpu_count > 4 else cpu_count))


def find_system_vulkan_driver_dir() -> Optional[str]:
    """
    Dynamically probes and returns the absolute directory path of the valid libvulkan.so driver installed on the system.
    Tier 1: Directly inspect /proc/self/maps Linux kernel virtual memory mapping (Ground Truth, 0% root required).
    Tier 2: Probe standard 64-bit / 32-bit Android HAL directories.
    """
    import sys
    import os
    import ctypes
    from pathlib import Path

    # Tier 1: Linux kernel process virtual memory map inspection (Ground Truth)
    try:
        handle = ctypes.CDLL("libvulkan.so")
        if hasattr(handle, "vkCreateInstance"):
            maps_path = Path("/proc/self/maps")
            if maps_path.is_file():
                with open(maps_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "libvulkan.so" in line:
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                real_path = parts[-1]
                                if os.path.isabs(real_path) and os.path.exists(real_path):
                                    return str(Path(real_path).parent.resolve())
    except Exception:
        pass

    # Tier 2: Search standard 64-bit vs 32-bit Android HAL directories
    is_64bit = sys.maxsize > 2**32

    if is_64bit:
        candidate_dirs = [
            Path("/system/lib64"),
            Path("/vendor/lib64"),
            Path("/apex/com.android.runtime/lib64"),
            Path("/system/vendor/lib64"),
        ]
    else:
        candidate_dirs = [
            Path("/system/lib"),
            Path("/vendor/lib"),
            Path("/apex/com.android.runtime/lib"),
            Path("/system/vendor/lib"),
        ]

    for root in candidate_dirs:
        so_path = root / "libvulkan.so"
        if so_path.is_file():
            try:
                h = ctypes.CDLL(str(so_path))
                if hasattr(h, "vkCreateInstance"):
                    return str(root.resolve())
            except Exception:
                continue

    # Tier 3: Simple file existence probe
    for root in candidate_dirs:
        if (root / "libvulkan.so").is_file():
            return str(root.resolve())

    return None


def get_vulkan_env(base_env: Optional[dict[str, str]] = None) -> dict[str, str]:
    """
    Returns environment variable dictionary for the AMEVA Vulkan runtime.
    Golden Link Order:
    1: System Vulkan driver directory (/system/lib64)
    2: Termux native C++ runtime (/data/data/com.termux/files/usr/lib)
    3: Bundled runtime library (~/.termux-llama/current/lib)
    4: Existing system and user environment paths
    """
    import os
    from pathlib import Path

    env = dict(base_env or os.environ)
    current_ld = env.get("LD_LIBRARY_PATH", "")

    # Dynamically probe verified Vulkan driver directory
    discovered_driver_dir = find_system_vulkan_driver_dir()

    ordered_dirs = []

    # Priority 1: Smartphone system Vulkan driver directory (/system/lib64)
    # Prevents Android 15 Bionic libunwindstack symbol collision (system liblzma binds first)
    if discovered_driver_dir and discovered_driver_dir not in ordered_dirs:
        ordered_dirs.append(discovered_driver_dir)
    elif os.path.isdir("/system/lib64") and "/system/lib64" not in ordered_dirs:
        ordered_dirs.append("/system/lib64")

    # Priority 2: Termux native C++ runtime
    termux_usr_lib = "/data/data/com.termux/files/usr/lib"
    if os.path.isdir(termux_usr_lib) and termux_usr_lib not in ordered_dirs:
        ordered_dirs.append(termux_usr_lib)

    # Priority 3: Bundled llama/runtime libraries
    llama_lib = str(Path.home() / ".termux-llama/current/lib")
    if os.path.isdir(llama_lib) and llama_lib not in ordered_dirs:
        ordered_dirs.append(llama_lib)

    # Priority 4: Append remaining paths without duplication
    existing_parts = [p for p in current_ld.split(":") if p]
    final_dirs = ordered_dirs + [p for p in existing_parts if p not in ordered_dirs]
    env["LD_LIBRARY_PATH"] = ":".join(final_dirs)

    # Mali GPU Quirks: Avoid infinite GEMM quantization loops on ARM Mali Valhall architectures
    try:
        from ..platform import detect_soc_environment
        soc = detect_soc_environment()
        if "mali" in str(soc.gpu_family).lower() or soc.vendor == "samsung":
            env.setdefault("GGML_VK_FORCE_MEDIUM_MATMUL", "1")
            env.setdefault("GGML_VK_DISABLE_F16", "1")
    except Exception:
        pass

    return env
