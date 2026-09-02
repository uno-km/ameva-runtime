"""
6-Modality Acceleration Adapters — 실제 엔진 바인딩 구현.

각 어댑터는 IVulkanConsumer Protocol 을 구현합니다.
engine=None 시 구성 정보만 반환하며, 실제 엔진 인스턴스가 주어지면
해당 엔진의 내부 설정(플래그, 스레드, GPU 레이어)을 실질적으로 조작합니다.

[오류 처리 원칙]
- 발생하는 모든 오류는 [ameva-vulkan-runtime:<AdapterName>] 태그로 logging 기록.
- engine=None 인 경우 AmevaRuntimeError 를 raise 하지 않고 구성 dict 만 반환.
- 실제 엔진 바인딩 실패 시 DriverQuirkViolationError 또는 AmevaRuntimeError 를 raise.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from ..doctor import DiagnosticReport
from ..exceptions import AmevaRuntimeError, DriverQuirkViolationError
from ..protocol import BindingResult, IVulkanConsumer

logger = logging.getLogger("ameva_vulkan_runtime.adapters")

# Qualcomm Vendor ID
_ADRENO_VENDOR_ID = 0x5143
# ARM Vendor ID
_MALI_VENDOR_ID = 0x13B5


def _is_vulkan_report(report: DiagnosticReport) -> bool:
    if report is None:
        return False
    return bool(
        report.overall_success
        or report.recommended_backend in ("vulkan", "vulkan_driver_only")
        or report.passed_stages >= 7
    )


def _make_cpu_fallback(module: str, report: DiagnosticReport, config: dict) -> BindingResult:
    """Vulkan 미지원 환경용 CPU NEON 폴백 BindingResult 생성."""
    logger.info(
        "[ameva-vulkan-runtime:%s] Vulkan 가속 미지원 — CPU NEON 모드로 전환합니다. "
        "원인: %s", module, report.device_name or "Vulkan 불가"
    )
    config["backend"] = "cpu_neon"
    return BindingResult(
        module=module, backend="cpu_neon", is_vulkan=False,
        device_name=report.device_name, vendor_id=report.vendor_id,
        config=config, status="BOUND_CPU",
    )


# ===========================================================================
# SttAdapter — termux-stt (whisper.cpp)
# ===========================================================================

class SttAdapter:
    """termux-stt (whisper.cpp / sherpa-onnx) Vulkan 가속 바인딩 어댑터.

    바인딩 전략:
    - whisper.cpp 가 `--gpu-layers` / `--vulkan` 플래그를 지원하면 WhisperEngine.config 에 주입.
    - 미지원 시 FP16 NEON 스레드 수를 Big-core 전용으로 최적화하여 CPU 최고 성능 보장.
    """

    module_name = "termux-stt"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": SttAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "encoder_fp16": True,
                "rtf_target": 0.28,
                "gpu_layers": 33,
                "vulkan_flag": True,
            })
            # 실제 WhisperEngine 인스턴스가 주어진 경우
            if engine is not None:
                try:
                    if hasattr(engine, "config"):
                        # WhisperEngine.config.extra 에 Vulkan 플래그 주입
                        engine.config.extra["gpu_layers"] = 33
                        engine.config.extra["use_vulkan"] = True
                        logger.info(
                            "[ameva-vulkan-runtime:SttAdapter] WhisperEngine.config.extra 에 "
                            "Vulkan 플래그 주입 완료 (device=%s, vendor=0x%04X)",
                            report.device_name, report.vendor_id
                        )
                    elif hasattr(engine, "set_vulkan"):
                        engine.set_vulkan(True, gpu_layers=33)
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:SttAdapter] engine 에 config 또는 set_vulkan 속성이 없습니다. "
                            "엔진 타입: %s — 구성 정보만 반환합니다.", type(engine).__name__
                        )
                except Exception as e:
                    logger.error(
                        "[ameva-vulkan-runtime:SttAdapter] 엔진 바인딩 중 오류: %s", e
                    )
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:SttAdapter] WhisperEngine Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=SttAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            # CPU 폴백: Big-core 전용 스레드 최적화
            import os
            cpu_cores = os.cpu_count() or 8
            optimal_threads = max(1, cpu_cores // 2)
            config["threads"] = optimal_threads
            config["rtf_target"] = 0.80
            if engine is not None and hasattr(engine, "threads"):
                engine.threads = optimal_threads
            return _make_cpu_fallback(SttAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:SttAdapter] 바인딩 해제.")


# ===========================================================================
# DiffusionAdapter — termux-diffusion (stable-diffusion.cpp)
# ===========================================================================

class DiffusionAdapter:
    """termux-diffusion (stable-diffusion.cpp) Vulkan 가속 바인딩 어댑터.

    바인딩 전략:
    - ameva Doctor 의 DiagnosticReport 에서 loader_path 를 추출하여
      sd-cli 의 CMake 플래그 생성에 사용합니다.
    - 기존 termux_diffusion.hardware._probe_vulkan_driver() 호출을 대체합니다.
    """

    module_name = "termux-diffusion"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": DiffusionAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vulkan_lib_path": report.loader_path,
                "unet_tiling": True,
                "sd_vulkan_flag": True,
                "coopmat_off": True,  # Adreno 830 NV coopmat2 버그 회피
            })

            if engine is not None:
                try:
                    # termux_diffusion 의 HardwareProfile 을 ameva report 로 패치
                    if hasattr(engine, "hw_profile"):
                        engine.hw_profile.vulkan_available = True
                        engine.hw_profile.vulkan_driver = type("GD", (), {
                            "library_path": report.loader_path,
                            "usable": True,
                        })()
                        logger.info(
                            "[ameva-vulkan-runtime:DiffusionAdapter] hw_profile.vulkan_driver 패치 완료: %s",
                            report.loader_path
                        )
                    elif hasattr(engine, "set_vulkan_lib"):
                        engine.set_vulkan_lib(report.loader_path)
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:DiffusionAdapter] engine 에 hw_profile 또는 "
                            "set_vulkan_lib 속성이 없습니다. 타입: %s", type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:DiffusionAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:DiffusionAdapter] sd.cpp Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=DiffusionAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["offload_to_cpu"] = True
            return _make_cpu_fallback(DiffusionAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:DiffusionAdapter] 바인딩 해제.")


# ===========================================================================
# BitnetAdapter — termux-bitnet (1.58-bit LLM)
# ===========================================================================

class BitnetAdapter:
    """termux-bitnet (BitNet 1.58-bit i2_s) Vulkan 가속 바인딩 어댑터.

    바인딩 전략:
    - BitNetEngine.config.n_gpu_layers 를 ameva report 기반으로 자동 설정.
    - Adreno 830 (vendorID=0x5143): 33레이어 GPU 오프로딩.
    - Mali-G78/G68 (vendorID=0x13B5): 128-byte 정렬 후 33레이어 오프로딩.
    """

    module_name = "termux-bitnet"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": BitnetAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            ngl = 33

            # Mali 128-byte 정렬 강제 여부
            mali_align_required = (report.vendor_id == _MALI_VENDOR_ID or
                                   "Mali" in report.device_name)

            config.update({
                "backend": "vulkan",
                "n_gpu_layers": ngl,
                "mali_128byte_align": mali_align_required,
                "flash_attn": True,
            })

            if engine is not None:
                try:
                    if hasattr(engine, "config"):
                        engine.config.n_gpu_layers = ngl
                        engine.config.flash_attn = True
                        logger.info(
                            "[ameva-vulkan-runtime:BitnetAdapter] config.n_gpu_layers=%d 설정 완료"
                            " (device=%s, mali_align=%s)", ngl, report.device_name, mali_align_required
                        )
                    else:
                        logger.warning(
                            "[ameva-vulkan-runtime:BitnetAdapter] engine.config 속성 없음. "
                            "타입: %s", type(engine).__name__
                        )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:BitnetAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:BitnetAdapter] BitNetEngine Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=BitnetAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            import os
            config["n_threads"] = max(1, (os.cpu_count() or 8) // 2)
            config["kernel"] = "neon_dotprod"
            if engine is not None and hasattr(engine, "config"):
                engine.config.n_gpu_layers = 0
            return _make_cpu_fallback(BitnetAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:BitnetAdapter] 바인딩 해제.")


# ===========================================================================
# LlamaCppAdapter — termux-llamacpp (GGUF LLM)
# ===========================================================================

class LlamaCppAdapter:
    """termux-llamacpp (llama.cpp GGUF) Vulkan 가속 바인딩 어댑터.

    바인딩 전략:
    - llama.cpp subprocess 호출 시 `-ngl 33` 및 `--device vulkan` 플래그를 동적 주입.
    - 엔진이 subprocess 빌더(cmd list) 또는 설정 객체 형태로 전달됩니다.
    """

    module_name = "termux-llamacpp"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": LlamaCppAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            import os
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            ngl = 33

            # Mali GPU 감지 여부
            is_mali = (report.vendor_id == _MALI_VENDOR_ID or "Mali" in report.device_name)

            config.update({
                "backend": "vulkan",
                "ngl": ngl,
                "device_flag": "vulkan",
                "flash_attn": True,
                "ctx_size": 2048,
                "threads": big_cores,
                "mali_align": is_mali,
            })

            if engine is not None:
                try:
                    # 1) engine 이 dict 형 config 인 경우
                    if isinstance(engine, dict):
                        engine.setdefault("ngl", ngl)
                        engine["device"] = "vulkan"
                        engine.setdefault("flash_attn", True)
                        engine.setdefault("threads", big_cores)
                    # 2) engine 이 engine.config (RuntimeConfig / EngineConfig) 객체를 가진 경우
                    elif hasattr(engine, "config"):
                        cfg = engine.config
                        if isinstance(cfg, dict):
                            cfg.setdefault("ngl", ngl)
                            cfg.setdefault("n_gpu_layers", ngl)
                            cfg["device"] = "vulkan"
                            cfg.setdefault("flash_attn", True)
                            cfg.setdefault("threads", big_cores)
                        else:
                            if hasattr(cfg, "n_gpu_layers"):
                                cfg.n_gpu_layers = ngl
                            if hasattr(cfg, "ngl"):
                                cfg.ngl = ngl
                            if hasattr(cfg, "device"):
                                cfg.device = "vulkan"
                            if hasattr(cfg, "flash_attn"):
                                cfg.flash_attn = True
                            if hasattr(cfg, "threads") and getattr(cfg, "threads", 0) == 0:
                                cfg.threads = big_cores
                    # 3) engine 이 단독 객체 형 config 인 경우
                    elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                        if hasattr(engine, "ngl") and getattr(engine, "ngl", 0) == 0:
                            engine.ngl = ngl
                        if hasattr(engine, "n_gpu_layers") and getattr(engine, "n_gpu_layers", 0) == 0:
                            engine.n_gpu_layers = ngl
                        if hasattr(engine, "device"):
                            engine.device = "vulkan"
                        if hasattr(engine, "threads") and getattr(engine, "threads", 0) == 0:
                            engine.threads = big_cores
                    # 4) engine 이 subprocess cmd list 인 경우
                    elif isinstance(engine, list):
                        if "-ngl" not in engine and "--n-gpu-layers" not in engine:
                            engine.extend(["-ngl", str(ngl)])
                        if "--device" not in engine and "-dev" not in engine:
                            engine.extend(["--device", "vulkan"])
                        if "-t" not in engine and "--threads" not in engine:
                            engine.extend(["-t", str(big_cores)])
                    logger.info(
                        "[ameva-vulkan-runtime:LlamaCppAdapter] -ngl %d -t %d --device vulkan 주입 완료"
                        " (device=%s, is_mali=%s)", ngl, big_cores, report.device_name, is_mali
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:LlamaCppAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:LlamaCppAdapter] llama.cpp Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=LlamaCppAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            import os
            cpu_cores = os.cpu_count() or 8
            big_cores = max(1, cpu_cores // 2)
            config["ngl"] = 0
            config["threads"] = big_cores
            if engine is not None:
                if isinstance(engine, dict):
                    engine["ngl"] = 0
                    engine.setdefault("threads", big_cores)
                elif hasattr(engine, "config"):
                    cfg = engine.config
                    if isinstance(cfg, dict):
                        cfg["ngl"] = 0
                        cfg["n_gpu_layers"] = 0
                        cfg.setdefault("threads", big_cores)
                    else:
                        if hasattr(cfg, "n_gpu_layers"):
                            cfg.n_gpu_layers = 0
                        if hasattr(cfg, "ngl"):
                            cfg.ngl = 0
                        if hasattr(cfg, "threads"):
                            cfg.threads = big_cores
                elif hasattr(engine, "ngl") or hasattr(engine, "n_gpu_layers"):
                    if hasattr(engine, "ngl"):
                        engine.ngl = 0
                    if hasattr(engine, "n_gpu_layers"):
                        engine.n_gpu_layers = 0
                    if hasattr(engine, "threads"):
                        engine.threads = big_cores
                elif isinstance(engine, list):
                    if "-t" not in engine and "--threads" not in engine:
                        engine.extend(["-t", str(big_cores)])
            return _make_cpu_fallback(LlamaCppAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:LlamaCppAdapter] 바인딩 해제.")


# ===========================================================================
# TtsAdapter — termux-tts (Piper / VITS HiFi-GAN)
# ===========================================================================

class TtsAdapter:
    """termux-tts (Piper TTS / VITS HiFi-GAN) Vulkan 가속 바인딩 어댑터."""

    module_name = "termux-tts"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": TtsAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "transposed_conv_vulkan": True,
                "latency_ms_target": 38.5,
                "fp16_vocoder": True,
            })

            if engine is not None:
                try:
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    if hasattr(engine, "fp16") and hasattr(engine, "device") and getattr(engine, "device") != "cpu":
                        engine.fp16 = True
                    logger.info(
                        "[ameva-vulkan-runtime:TtsAdapter] Piper/VITS Vulkan 바인딩 완료"
                        " (device=%s)", report.device_name
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:TtsAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:TtsAdapter] TTS Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=TtsAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["latency_ms_target"] = 115.0
            return _make_cpu_fallback(TtsAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:TtsAdapter] 바인딩 해제.")


# ===========================================================================
# VisionAdapter — termux-vision (LLaVA ViT / YOLO)
# ===========================================================================

class VisionAdapter:
    """termux-vision (LLaVA ViT / SmolVLM / YOLO) Vulkan 가속 바인딩 어댑터.

    [주의] termux-vision 은 자체 `libfast_cv_vk.so` Vulkan 스택을 보유합니다.
    공존 전략: ameva Doctor 의 `loader_path` 를 vision 의 `_vk_lib_path` 와 비교하여
    동일한 ICD 경로를 사용하도록 유도합니다. 충돌 시 WARNING 을 기록하고 vision 자체 스택을 유지합니다.
    """

    module_name = "termux-vision"

    @staticmethod
    def bind(engine: Any, report: DiagnosticReport) -> BindingResult:
        is_vk = _is_vulkan_report(report)
        config: dict = {
            "module": VisionAdapter.module_name,
            "device_name": report.device_name,
            "vendor_id": report.vendor_id,
        }

        if is_vk:
            config.update({
                "backend": "vulkan",
                "vit_acceleration": True,
                "patch_embedding_vulkan": True,
                "ameva_loader_path": report.loader_path,
            })

            if engine is not None:
                try:
                    # termux_vision.csrc.backend 모듈의 전역 상태 확인
                    try:
                        from termux_vision.csrc import backend as vk_backend
                        vision_lib_path = getattr(vk_backend, "_vk_lib_path", None)
                        if vision_lib_path and vision_lib_path != report.loader_path:
                            logger.warning(
                                "[ameva-vulkan-runtime:VisionAdapter] Vulkan ICD 경로 불일치 감지. "
                                "ameva=%s | vision=%s. "
                                "termux-vision 자체 libfast_cv_vk.so 스택을 유지합니다.",
                                report.loader_path, vision_lib_path
                            )
                        else:
                            logger.info(
                                "[ameva-vulkan-runtime:VisionAdapter] Vulkan ICD 경로 일치 확인: %s",
                                report.loader_path
                            )
                    except ImportError:
                        logger.warning(
                            "[ameva-vulkan-runtime:VisionAdapter] termux_vision 패키지를 import 할 수 없습니다. "
                            "vision 이 설치되어 있는지 확인하세요."
                        )

                    if hasattr(engine, "device"):
                        engine.device = "vulkan"
                    if hasattr(engine, "use_vulkan"):
                        engine.use_vulkan = True
                    logger.info(
                        "[ameva-vulkan-runtime:VisionAdapter] LLaVA/YOLO Vulkan ViT 바인딩 완료."
                    )
                except Exception as e:
                    logger.error("[ameva-vulkan-runtime:VisionAdapter] 바인딩 오류: %s", e)
                    raise AmevaRuntimeError(
                        f"[ameva-vulkan-runtime:VisionAdapter] Vision Vulkan 바인딩 실패: {e}"
                    ) from e

            return BindingResult(
                module=VisionAdapter.module_name, backend="vulkan", is_vulkan=True,
                device_name=report.device_name, vendor_id=report.vendor_id,
                config=config, status="BOUND",
            )
        else:
            config["vit_acceleration"] = False
            return _make_cpu_fallback(VisionAdapter.module_name, report, config)

    @staticmethod
    def unbind(engine: Any = None) -> None:
        logger.info("[ameva-vulkan-runtime:VisionAdapter] 바인딩 해제.")
