"""
Termux-Train Command-Line Interface (CLI)
=========================================
Unified edge training CLI supporting pure GPU-accelerated on-device fine-tuning.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from .backends.base import TrainingConfig
from .core import TrainingSession
from .exceptions import TermuxTrainError
from .utils.hardware import probe_hardware
from .utils.monitor import ResourceMonitor

logger = logging.getLogger("termux_train.cli")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def cmd_doctor(args: argparse.Namespace) -> int:
    """Probes edge device hardware readiness for on-device training."""
    print("=" * 65)
    print("  Termux-Train: On-Device Hardware & Training Readiness Probe")
    print("=" * 65)
    hw = probe_hardware()
    print(f"  Target Device        : {hw.device_name}")
    print(f"  Vulkan Available     : {'YES' if hw.has_vulkan else 'NO'}")
    print(f"  Vulkan Loader Path   : {hw.vulkan_lib_path or 'None'}")
    print(f"  Unified Memory (UMA) : {'YES (Zero-Copy Compatible)' if hw.is_unified_memory else 'NO'}")
    print(f"  Total RAM            : {hw.total_ram_mb} MB")
    print(f"  Available RAM        : {hw.available_ram_mb} MB")

    temp_c = ResourceMonitor.read_battery_temperature_c()
    temp_str = f"{temp_c:.1f}°C" if temp_c is not None else "N/A (Non-Android sysfs)"
    print(f"  Battery / AP Temp    : {temp_str}")
    print("-" * 65)

    if not hw.has_vulkan:
        print("[WARNING] Vulkan driver not detected. Pure GPU training will require native loader binding.")
    else:
        print("[SUCCESS] Vulkan driver validated for GPU-accelerated backpropagation.")
    print("=" * 65)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Dispatches a training session for the specified modality."""
    config = TrainingConfig(
        model_path=args.model,
        dataset_path=args.train_data,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        epochs=args.epochs,
        threads=args.threads,
        context_length=args.ctx,
        strict_gpu=not args.allow_cpu_fallback,
    )

    try:
        session = TrainingSession(modality=args.modality, config=config)
        result = session.run()
        print("\n" + "=" * 65)
        print("  [TRAINING COMPLETE] On-Device Adaptation Succeeded")
        print("=" * 65)
        print(f"  Artifact Path : {result.output_artifact_path}")
        print(f"  Total Steps   : {result.total_steps}")
        print(f"  Final Loss    : {result.final_loss:.4f}")
        print(f"  Duration      : {result.duration_seconds:.1f}s")
        print("=" * 65)
        return 0
    except TermuxTrainError as err:
        print(f"\n[FATAL ERROR] {err.message}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"\n[UNEXPECTED CRASH] {err}", file=sys.stderr)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="termux-train",
        description="Edge-Native On-Device Training Framework (Zero-Silent-Fallback Standard)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    sub_doc = subparsers.add_parser("doctor", help="Inspect hardware and GPU training readiness")
    sub_doc.set_defaults(func=cmd_doctor)

    # 5 Modalities: llm, stt, tts, diff, vision
    for mod in ["llm", "stt", "tts", "diff", "vision"]:
        sub = subparsers.add_parser(mod, help=f"Run on-device training for modality '{mod}'")
        sub.add_argument("--model", required=True, help="Path to base model weights file")
        sub.add_argument("--train-data", required=True, help="Path to training dataset/corpus file")
        sub.add_argument("--output-dir", default="./train_output", help="Directory to save LoRA adapters")
        sub.add_argument("--lora-r", type=int, default=8, help="LoRA rank dimension (default: 8)")
        sub.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling alpha (default: 16)")
        sub.add_argument("--batch-size", type=int, default=1, help="Micro batch size (default: 1)")
        sub.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps (default: 16)")
        sub.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
        sub.add_argument("--threads", type=int, default=4, help="CPU thread count (default: 4)")
        sub.add_argument("--ctx", type=int, default=256, help="Context sequence length (default: 256)")
        sub.add_argument("--allow-cpu-fallback", action="store_true", help="Allow fallback to CPU if GPU fails (Strict GPU is default)")
        sub.set_defaults(func=cmd_train, modality=mod)

    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
