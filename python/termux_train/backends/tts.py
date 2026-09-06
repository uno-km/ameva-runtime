"""
Termux-Train TTS Backend (Speaker Adaptation) — Planned Stage 3
==============================================================
"""
from __future__ import annotations

from .base import BaseTrainer, TrainingConfig, TrainingResult
from ..exceptions import TermuxTrainError


class TtsTrainer(BaseTrainer):
    """Piper/Kokoro Speaker Embedding fine-tuning backend (Roadmap Stage 3)."""

    def resolve_binary_path(self) -> str:
        raise TermuxTrainError(
            "TTS (보이스 어댑터) 트레이너는 로드맵 제3단계 예정 항목입니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def train(self) -> TrainingResult:
        raise TermuxTrainError(
            "TTS (보이스 어댑터) 학습 파이프라인은 제2단계(STT) 완료 후 활성화됩니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )
