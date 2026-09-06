"""
Termux-Train Diffusion Backend (Latent Caching & U-Net LoRA) — Planned Stage 4
=============================================================================
"""
from __future__ import annotations

from .base import BaseTrainer, TrainingConfig, TrainingResult
from ..exceptions import TermuxTrainError


class DiffusionTrainer(BaseTrainer):
    """Stable Diffusion Latent-Cached U-Net LoRA fine-tuning backend (Roadmap Stage 4)."""

    def resolve_binary_path(self) -> str:
        raise TermuxTrainError(
            "Diffusion (U-Net LoRA) 트레이너는 로드맵 제4단계 예정 항목입니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )

    def build_command(self) -> list[str]:
        raise NotImplementedError

    def train(self) -> TrainingResult:
        raise TermuxTrainError(
            "Diffusion (U-Net LoRA) 학습 파이프라인은 제3단계(TTS) 완료 후 활성화됩니다.",
            error_code="STAGE_NOT_YET_IMPLEMENTED",
        )
