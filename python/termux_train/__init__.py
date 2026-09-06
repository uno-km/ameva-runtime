"""
Termux-Train: Edge-Native On-Device Training Framework
======================================================
Pure GPU-accelerated training pipeline for mobile edge devices (Android/Termux).
"""
from .backends.base import CheckpointInfo, TrainingConfig, TrainingResult
from .core import TrainingSession
from .exceptions import (
    CheckpointCorruptionError,
    ExecutionEnvironmentError,
    GpuOperatorNotSupportedError,
    TermuxTrainError,
    ThermalThrottledError,
    TrainingOutOfMemoryError,
)

__version__ = "0.1.0"

__all__ = [
    "TrainingSession",
    "TrainingConfig",
    "TrainingResult",
    "CheckpointInfo",
    "TermuxTrainError",
    "ExecutionEnvironmentError",
    "GpuOperatorNotSupportedError",
    "TrainingOutOfMemoryError",
    "ThermalThrottledError",
    "CheckpointCorruptionError",
]
