"""
Termux-Train Exception Hierarchy
================================
Zero-Silent-Fallback and Fail-Fast exception standards for edge-native training.
"""
from __future__ import annotations
from typing import List, Optional


class TermuxTrainError(Exception):
    """Base exception for all termux-train runtime errors."""

    def __init__(self, message: str, error_code: str = "TRAIN_INTERNAL_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class ExecutionEnvironmentError(TermuxTrainError):
    """Raised when critical hardware drivers or native binaries are missing."""

    def __init__(self, missing_component: str, path_searched: str):
        message = f"필수 실행 환경 컴포넌트를 찾을 수 없습니다: '{missing_component}' (검색 경로: {path_searched})"
        super().__init__(message, error_code="ENV_COMPONENT_MISSING")


class GpuOperatorNotSupportedError(TermuxTrainError):
    """Raised when the GPU backend rejects a required backpropagation or optimizer operator."""

    def __init__(self, operator_name: str, backend: str, unsupported_reasons: Optional[List[str]] = None):
        reasons_str = f" | 세부 사유: {', '.join(unsupported_reasons)}" if unsupported_reasons else ""
        message = (
            f"GPU 백엔드('{backend}')가 역전파/옵티마이저 연산자 '{operator_name}'를 지원하지 않습니다.{reasons_str} "
            "침묵 폴백이 금지되어 있으므로 학습을 즉시 중단합니다."
        )
        super().__init__(message, error_code="GPU_OPERATOR_UNSUPPORTED")


class TrainingOutOfMemoryError(TermuxTrainError):
    """Raised when system RAM or VRAM approaches critical limits threatening LMK kill."""

    def __init__(self, current_mb: int, limit_mb: int, stage: str):
        message = (
            f"메모리 한계 임계치에 도달했습니다 (현재: {current_mb}MB / 제한: {limit_mb}MB, 단계: {stage}). "
            "Android Low Memory Killer(LMK) 강제 종료를 방지하기 위해 안전하게 중단합니다."
        )
        super().__init__(message, error_code="TRAIN_OOM_PREVENTED")


class ThermalThrottledError(TermuxTrainError):
    """Raised when hardware temperature exceeds the safe boundary for continuous training."""

    def __init__(self, current_temp_c: float, max_safe_temp_c: float):
        message = (
            f"디바이스 배터리/AP 온도가 안전 임계치를 초과했습니다 (현재: {current_temp_c:.1f}°C / 안전치: {max_safe_temp_c:.1f}°C). "
            "하드웨어 손상 및 배터리 과열 방지를 위해 학습이 중단되었습니다."
        )
        super().__init__(message, error_code="THERMAL_LIMIT_EXCEEDED")


class CheckpointCorruptionError(TermuxTrainError):
    """Raised when a saved training checkpoint fails integrity validation."""

    def __init__(self, checkpoint_path: str, reason: str):
        message = f"체크포인트 무결성 검증 실패: '{checkpoint_path}' (원인: {reason})"
        super().__init__(message, error_code="CHECKPOINT_CORRUPTED")
