"""
Termux-Train STT Backend (Whisper LoRA) — Planned Stage 2
=========================================================
"""
from __future__ import annotations

from .base import BaseTrainer, TrainingConfig, TrainingResult
from ..exceptions import TermuxTrainError


class WhisperTrainer(BaseTrainer):
    """Whisper Decoder LoRA fine-tuning backend (Roadmap Stage 2)."""

    def resolve_binary_path(self) -> str:
        raise TermuxTrainError(
            "STT (Whisper LoRA) 트레이너는 로드맵 제2단계 예정 항목입니다 (현재: 제1단계 LLM 진행 중).",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def train(self) -> TrainingResult:
        raise TermuxTrainError(
            "STT (Whisper LoRA) 학습 파이프라인은 제1단계(LLM) 검증 완료 후 활성화됩니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )
