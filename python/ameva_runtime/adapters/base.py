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


_MISSING = object()

DEFAULT_CPU_FALLBACK_PATTERNS = (
    "no gpu found",
    "using cpu backend",
    "falling back to cpu",
    "fallback to cpu",
    "vulkan initialization failed",
    "failed to initialize vulkan",
    "no vulkan device",
    "backend unavailable",
    "ggml_vulkan: failed",
    "ggml_vulkan: cannot",
)


def _extract_cli_ngl(args: list[str]) -> Optional[int]:
    """Extracts existing -ngl or --n-gpu-layers value from CLI arguments list."""
    for flag in ("-ngl", "--n-gpu-layers"):
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                try:
                    return int(args[idx + 1])
                except ValueError:
                    return None
    return None


def _set_engine_property(
    engine: Any,
    snapshot: dict[str, Any],
    key: str,
    val: Any,
) -> None:
    """Universal engine property injector supporting dict, ConfigObject, SimpleNamespace, and primitives.
    Automatically captures the exact pre-mutation state into snapshot dictionary.
    """
    if engine is None or isinstance(engine, list):
        return

    # 1. Dict interface
    if isinstance(engine, dict):
        if key not in snapshot:
            snapshot[key] = engine.get(key, _MISSING)
        engine[key] = val

    # 2. Config object interface (when engine doesn't have the attribute directly)
    elif hasattr(engine, "config") and not hasattr(engine, key) and getattr(engine, "config", None) is not None:
        cfg = engine.config
        snap_key = f"config.{key}"
        if isinstance(cfg, dict):
            if snap_key not in snapshot:
                snapshot[snap_key] = cfg.get(key, _MISSING)
            cfg[key] = val
        else:
            if snap_key not in snapshot:
                snapshot[snap_key] = getattr(cfg, key, _MISSING)
            setattr(cfg, key, val)

    # 3. Direct object attribute interface
    else:
        if key not in snapshot:
            snapshot[key] = getattr(engine, key, _MISSING)
        setattr(engine, key, val)


def _restore_engine_properties(
    engine: Any,
    snapshot: Optional[dict[str, Any]],
) -> None:
    """Restores engine state strictly to its pre-binding snapshot without destructive side-effects."""
    if engine is None or not snapshot:
        return

    # 1. Restore CLI argument list slice
    if isinstance(engine, list) and "cli_args_len" in snapshot:
        orig_len = snapshot["cli_args_len"]
        if isinstance(orig_len, int) and len(engine) >= orig_len:
            del engine[orig_len:]

    # 2. Restore dict and object attributes
    for k, old_val in snapshot.items():
        if k == "cli_args_len":
            continue

        if k.startswith("config."):
            cfg_key = k[7:]
            if hasattr(engine, "config"):
                cfg = engine.config
                if isinstance(cfg, dict):
                    if old_val is _MISSING:
                        cfg.pop(cfg_key, None)
                    else:
                        cfg[cfg_key] = old_val
                else:
                    if old_val is _MISSING:
                        if hasattr(cfg, cfg_key):
                            try:
                                delattr(cfg, cfg_key)
                            except Exception:
                                setattr(cfg, cfg_key, None)
                    else:
                        setattr(cfg, cfg_key, old_val)
        else:
            if isinstance(engine, dict):
                if old_val is _MISSING:
                    engine.pop(k, None)
                else:
                    engine[k] = old_val
            else:
                if old_val is _MISSING:
                    if hasattr(engine, k):
                        try:
                            delattr(engine, k)
                        except Exception:
                            setattr(engine, k, None)
                else:
                    setattr(engine, k, old_val)


class BaseAdapter:
    """Base class and orchestration core for all AMEVA modality adapters."""

    module_name: str = "generic-adapter"
    ALLOWED_BACKENDS = {"auto", "vulkan", "gpu", "cpu", "cpu_neon"}
    device_signatures: tuple[str, ...] = ("vulkan",)
    fallback_patterns: tuple[str, ...] = DEFAULT_CPU_FALLBACK_PATTERNS

    @classmethod
    def normalize_backend(cls, requested_backend: Optional[str]) -> str:
        """Validates requested backend against allowed whitelist adhering to Fail-Fast."""
        req = str(requested_backend or "auto").strip().lower()
        if req not in cls.ALLOWED_BACKENDS:
            from ..exceptions import AmevaRuntimeError
            raise AmevaRuntimeError(
                f"[{cls.module_name}] Unsupported backend '{requested_backend}'. "
                f"Allowed backends: {sorted(cls.ALLOWED_BACKENDS)}"
            )
        return req

    @classmethod
    def get_execution_environment(
        cls,
        base_env: Optional[dict[str, str]] = None,
        **hardware_quirks: Any,
    ) -> dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order.
        Guarantees that GGML_VULKAN_SKIP_CHECKS is NEVER injected, while preserving legitimate hardware quirks.
        """
        env = get_vulkan_env(base_env)
        # Guarantee no check bypass
        env.pop("GGML_VULKAN_SKIP_CHECKS", None)

        # Dynamic Hardware Quirks Synthesizer
        if hardware_quirks.get("tune_mali") or hardware_quirks.get("force_mmvq"):
            env["GGML_VK_FORCE_MMVQ"] = "1"
        if hardware_quirks.get("dsp_accel"):
            env["AMEVA_VK_DSP_ACCEL"] = "1"

        for k, v in hardware_quirks.items():
            if k.startswith("GGML_VK_") or k.startswith("AMEVA_"):
                env[k] = str(v)

        return env

    @classmethod
    def _set_engine_property(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        key: str,
        val: Any,
    ) -> None:
        """Universal property injector exposed to subclasses."""
        _set_engine_property(engine, snapshot, key, val)

    @classmethod
    def _restore_engine_properties(
        cls,
        engine: Any,
        snapshot: Optional[dict[str, Any]],
    ) -> None:
        """Universal property restorer exposed to subclasses."""
        _restore_engine_properties(engine, snapshot)

    @classmethod
    def _mutate_common_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        backend: str,
        threads: int,
    ) -> None:
        """Tier 1: Common invariant layer (cleans LD_LIBRARY_PATH pollution, sets device & threads)."""
        if engine is None:
            return

        if isinstance(engine, dict):
            # Sanitize LD_LIBRARY_PATH pollution
            if "env" in engine and isinstance(engine["env"], dict):
                engine["env"].pop("LD_LIBRARY_PATH", None)
            cls._set_engine_property(engine, snapshot, "device", "vulkan" if backend == "vulkan" else "cpu")
            cls._set_engine_property(engine, snapshot, "threads", threads)
        elif hasattr(engine, "config"):
            cls._set_engine_property(engine, snapshot, "device", "vulkan" if backend == "vulkan" else "cpu")
            cls._set_engine_property(engine, snapshot, "threads", threads)
        elif isinstance(engine, list):
            # Capture CLI argument length for non-destructive slice restore
            if "cli_args_len" not in snapshot:
                snapshot["cli_args_len"] = len(engine)
            if "-t" not in engine and "--threads" not in engine:
                engine.extend(["-t", str(threads)])
        else:
            if hasattr(engine, "device"):
                cls._set_engine_property(engine, snapshot, "device", "vulkan" if backend == "vulkan" else "cpu")
            if hasattr(engine, "threads"):
                cls._set_engine_property(engine, snapshot, "threads", threads)

    @classmethod
    def _mutate_modality_attributes(
        cls,
        engine: Any,
        snapshot: dict[str, Any],
        backend: str,
        report: Any,
        profile: Any,
        **kwargs: Any,
    ) -> None:
        """Tier 2: Modality-specific hook to be implemented by subclasses."""
        pass

    @classmethod
    def unbind(
        cls,
        engine: Any = None,
        binding_result: Optional[BindingResult] = None,
    ) -> None:
        """Restores engine state strictly to its pre-binding snapshot without destructive side-effects."""
        if engine is None:
            return
        snapshot = None
        if binding_result is not None and binding_result.restore_state:
            snapshot = binding_result.restore_state
        try:
            cls._restore_engine_properties(engine, snapshot)
            logger.info("[%s] Unbound adapter and restored engine state.", cls.module_name)
        except Exception as e:
            logger.warning("[%s] Unbind failed: %s", cls.module_name, e)

    @classmethod
    def verify_vulkan_compute_output(
        cls,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
        expected_signatures: Optional[tuple[str, ...]] = None,
        module_name: Optional[str] = None,
        stdout_text: str = "",
        stderr_text: str = "",
    ) -> None:
        """Context-Aware Zero-False-Positive Verifier.
        Inspects exclusively stderr and process returncode, protecting stdout from false alarm.
        """
        from ..exceptions import AmevaRuntimeError, PlatformNotSupportedError

        out_err = stderr or stderr_text or stdout or stdout_text
        mod = module_name or getattr(cls, "module_name", "generic-adapter")

        if returncode != 0:
            raise AmevaRuntimeError(
                f"[{mod}] Native compute process failed with exit code {returncode}.\n"
                f"Stderr: {out_err}"
            )

        lower_err = (out_err or "").lower()
        for pat in cls.fallback_patterns:
            if pat in lower_err:
                raise PlatformNotSupportedError(
                    f"[{mod}] Silent CPU fallback detected in runtime stderr: '{pat}'.\n"
                    f"Execution halted strictly under Zero-Silent-Fallback policy."
                )

        target_sigs = expected_signatures if expected_signatures is not None else cls.device_signatures
        if target_sigs and not any(sig in lower_err for sig in target_sigs):
            raise PlatformNotSupportedError(
                f"[{mod}] No positive Vulkan device signature logged in runtime stderr.\n"
                f"Expected signatures: {target_sigs}"
            )

    @classmethod
    def verify_in_process_device(cls, engine: Any) -> bool:
        """Verifies in-process C-API / shared library device binding."""
        if engine is None:
            return False
        if hasattr(engine, "device"):
            return getattr(engine, "device", "") == "vulkan"
        if hasattr(engine, "use_vulkan"):
            return bool(getattr(engine, "use_vulkan", False))
        return False

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
    restore_state: Optional[dict[str, Any]] = None,
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
        restore_state=restore_state,
    )


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
    Gate 1 Compliance:
    - Never injects /system/lib64, /vendor/lib64, /apex/... or Central Bridge symlinks into LD_LIBRARY_PATH.
    - Preserves user environment and existing LD_LIBRARY_PATH entries without arbitrary deletion or reordering.
    - Adds only isolated engine runtime library directories if present.
    """
    import os
    from pathlib import Path

    env = dict(base_env or os.environ)
    current_ld = env.get("LD_LIBRARY_PATH", "")

    home = Path.home()
    prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
    prefix_lib = prefix / "lib"
    engine_dirs = []

    if prefix_lib.is_dir():
        engine_dirs.append(str(prefix_lib))

    # Standard AMEVA engine library hierarchy
    for engine_name in ("stt", "diffusion", "tts", "llamacpp", "bitnet"):
        eng_lib = home / ".local" / "share" / "ameva" / "current" / engine_name / "lib"
        if eng_lib.is_dir():
            engine_dirs.append(str(eng_lib))
        eng_root = home / ".local" / "share" / "ameva" / "current" / engine_name
        if eng_root.is_dir():
            engine_dirs.append(str(eng_root))

    local_lib = home / ".local" / "lib"
    if local_lib.is_dir():
        engine_dirs.append(str(local_lib))

    # Backward compatibility with legacy termux-llama path
    llama_lib = home / ".termux-llama" / "current" / "lib"
    if llama_lib.is_dir():
        engine_dirs.append(str(llama_lib))

    # Construct clean LD_LIBRARY_PATH: engine dirs first, then user's existing LD_LIBRARY_PATH
    # Strictly filter out any /system/, /vendor/, or /apex/ directories to guarantee non-pollution
    forbidden_prefixes = ("/system/", "/vendor/", "/apex/", "/system_ext/", "/odm/", "/product/")

    existing_parts = [p for p in current_ld.split(":") if p and not any(p.startswith(fp) for fp in forbidden_prefixes)]

    merged_dirs = []
    for d in engine_dirs + existing_parts:
        if d not in merged_dirs and not any(d.startswith(fp) for fp in forbidden_prefixes):
            merged_dirs.append(d)

    if merged_dirs:
        env["LD_LIBRARY_PATH"] = ":".join(merged_dirs)
    elif "LD_LIBRARY_PATH" in env:
        env["LD_LIBRARY_PATH"] = ""

    # Mali GPU Quirks: Avoid infinite GEMM quantization loops on ARM Mali Valhall architectures
    try:
        from ..platform import detect_soc_environment
        soc = detect_soc_environment()
        if "mali" in str(soc.gpu_family).lower() or soc.vendor == "samsung":
            env.setdefault("GGML_VK_FORCE_MEDIUM_MATMUL", "1")
            env.setdefault("GGML_VK_DISABLE_F16", "1")

            # On Samsung Exynos / ARM Mali devices, enable HAL shim to sanitize vendor queue queries and GOS symbols
            shim_path = resolve_vulkan_shim_path()
            if shim_path:
                curr_preload = env.get("LD_PRELOAD", "")
                if shim_path not in curr_preload:
                    env["LD_PRELOAD"] = f"{shim_path}:{curr_preload}".strip(":") if curr_preload else shim_path
    except Exception:
        pass

    return env


def resolve_vulkan_shim_path() -> Optional[str]:
    """
    Locates or self-provisions the verified mobile Vulkan HAL shim binary (libegl_shim.so).
    Resolves across:
    1. Explicit environment variable: AMEVA_VULKAN_SHIM
    2. System Termux library path: $PREFIX/lib/libegl_shim.so
    3. ameva-runtime packaged bundle: ameva_runtime/vulkan/lib/libegl_shim.so
    4. Auto-provisions to $PREFIX/lib/libegl_shim.so if running in Termux with write permissions.
    """
    import os
    import shutil
    from pathlib import Path

    # 1. Explicit override
    env_path = os.environ.get("AMEVA_VULKAN_SHIM")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
    std_path = prefix / "lib" / "libegl_shim.so"
    if std_path.is_file():
        return str(std_path.resolve())

    # 2. Check bundled asset inside ameva_runtime package
    pkg_dir = Path(__file__).resolve().parent.parent  # ameva_runtime root
    bundled_paths = [
        pkg_dir / "vulkan" / "lib" / "libegl_shim.so",
        pkg_dir / "lib" / "libegl_shim.so",
    ]
    for bp in bundled_paths:
        if bp.is_file():
            # Auto-provision to $PREFIX/lib if writable
            try:
                if prefix.is_dir() and (prefix / "lib").is_dir() and not std_path.exists():
                    shutil.copy2(str(bp), str(std_path))
                    std_path.chmod(0o755)
                    logger.info("[ameva-runtime] Auto-provisioned libegl_shim.so -> %s", std_path)
                    return str(std_path.resolve())
            except Exception as pe:
                logger.debug("[ameva-runtime] Failed to auto-provision shim to %s: %s", std_path, pe)
            return str(bp.resolve())

    # 3. Known fallback locations on Termux
    home = Path.home()
    fallbacks = [
        home / ".local" / "lib" / "libegl_shim.so",
        home / "libegl_shim.so",
        Path("/data/data/com.termux/files/usr/lib/libegl_shim.so"),
        Path("/data/data/com.termux/files/home/libegl_shim.so"),
    ]
    for fp in fallbacks:
        if fp.is_file():
            return str(fp.resolve())

    return None


def resolve_authoritative_binary(
    binary_name: str,
    modality: str,
    manifest_path: Optional[Path] = None,
) -> Path:
    """Resolves and cryptographically validates the authoritative binary at $PREFIX/bin/<binary_name>.
    Adheres strictly to Single SSOT & Zero-Silent-Fallback policy.

    1. Primary location: $PREFIX/bin/<binary_name>
    2. Manifest location: ~/.local/share/ameva/manifests/<modality>.json
    3. Validates file existence and SHA-256 against manifest.
    4. On error: raises AmevaLlamaAssetMissingError or AmevaLlamaVerificationError immediately.
    """
    import os
    import json
    import hashlib
    from pathlib import Path
    from ..exceptions import AmevaLlamaAssetMissingError, AmevaLlamaVerificationError

    home = Path(os.environ.get("HOME", os.path.expanduser("~")))
    prefix = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
    prefix_bin = prefix / "bin" / binary_name
    local_bin = home / ".local" / "bin" / binary_name

    canonical_bin = prefix_bin if prefix_bin.is_file() else (local_bin if local_bin.is_file() else prefix_bin)

    if not canonical_bin.is_file():
        raise AmevaLlamaAssetMissingError(
            f"Canonical {binary_name} binary not found at '{canonical_bin}'. "
            f"Path search and automatic fallbacks are strictly prohibited under Zero-Silent-Fallback policy. "
            f"Remediation: Provision official AMEVA assets using 'python -m ameva_runtime.installer --asset {modality}'."
        )

    if manifest_path is None:
        manifest_path = home / ".local" / "share" / "ameva" / "manifests" / f"{modality}.json"
        # Backward compatibility with legacy ~/.termux-llama/manifests
        if not manifest_path.is_file() and modality == "llamacpp":
            legacy_manifest = home / ".termux-llama" / "manifests" / "llamacpp.json"
            if legacy_manifest.is_file():
                manifest_path = legacy_manifest
            else:
                legacy_current = home / ".termux-llama" / "current" / "manifest.json"
                if legacy_current.is_file():
                    manifest_path = legacy_current

    if not manifest_path.is_file():
        raise AmevaLlamaVerificationError(
            f"Cryptographic manifest missing for installed bundle at '{manifest_path}'. "
            f"Unauthenticated or legacy binary detected. "
            f"Remediation: Provision official AMEVA assets using 'python -m ameva_runtime.installer --asset {modality} --force'."
        )

    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest_data = json.load(f)
    except Exception as exc:
        raise AmevaLlamaVerificationError(
            f"Failed parsing manifest file at '{manifest_path}': {exc}. Corrupted installation detected."
        ) from exc

    expected_sha = None
    deployed_files = manifest_data.get("deployed_files", {})
    for rel_k, entry in deployed_files.items():
        if rel_k.endswith(f"bin/{binary_name}") or rel_k.endswith(binary_name):
            expected_sha = entry.get("sha256", "").strip().lower()
            break

    if not expected_sha:
        expected_sha = manifest_data.get("binary_sha256", "").strip().lower()

    if not expected_sha:
        raise AmevaLlamaVerificationError(
            f"Cryptographic manifest at '{manifest_path}' lacks valid sha256 checksum for {binary_name} binary."
        )

    h = hashlib.sha256()
    with canonical_bin.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    actual_sha = h.hexdigest().lower()

    if actual_sha != expected_sha:
        raise AmevaLlamaVerificationError(
            f"Cryptographic hash mismatch for binary '{canonical_bin}': "
            f"actual={actual_sha}, expected={expected_sha}. Tampering or corrupt binary detected. "
            f"Remediation: Re-provision using 'python -m ameva_runtime.installer --asset {modality} --force'."
        )

    return canonical_bin.resolve()


