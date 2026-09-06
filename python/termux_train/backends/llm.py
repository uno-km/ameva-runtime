"""
Termux-Train LLM LoRA Backend (llama-finetune)
==============================================
Pure GPU-accelerated GGUF LoRA fine-tuning backend utilizing llama.cpp native tools.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

from .base import BaseTrainer, CheckpointInfo, TrainingConfig, TrainingResult
from ..exceptions import ExecutionEnvironmentError, GpuOperatorNotSupportedError, TermuxTrainError

logger = logging.getLogger("termux_train.backends.llm")


class LlamaTrainer(BaseTrainer):
    """Native GGUF LoRA trainer using llama.cpp's llama-finetune binary."""

    CANDIDATE_BINARIES = [
        "llama-finetune",
        "llama-train",
        "/data/data/com.termux/files/usr/bin/llama-finetune",
        "/data/data/com.termux/files/home/.local/bin/llama-finetune",
        str(Path.home() / ".local" / "bin" / "llama-finetune"),
        str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-finetune"),
    ]

    def resolve_binary_path(self) -> str:
        """Finds verified physical llama-finetune binary."""
        # 1. Explicit env override
        env_bin = os.environ.get("AMEVA_LLAMA_FINETUNE_BIN")
        if env_bin and os.path.isfile(env_bin) and (os.access(env_bin, os.X_OK) or os.name == "nt"):
            return os.path.abspath(env_bin)

        # 2. Candidate paths and PATH lookup
        for cand in self.CANDIDATE_BINARIES:
            found = shutil.which(cand) if not os.path.isabs(cand) else cand
            if found and os.path.isfile(found) and (os.access(found, os.X_OK) or os.name == "nt"):
                return os.path.abspath(found)

        raise ExecutionEnvironmentError(
            missing_component="llama-finetune (llama.cpp native training binary)",
            path_searched=", ".join(self.CANDIDATE_BINARIES),
        )

    def get_execution_environment(self) -> Dict[str, str]:
        """Provides verified execution environment adhering to Golden Link Order."""
        env = os.environ.copy()
        if self.hardware.has_vulkan and self.hardware.vulkan_lib_path:
            loader_dir = os.path.dirname(self.hardware.vulkan_lib_path)
            ld_path = env.get("LD_LIBRARY_PATH", "")
            if loader_dir not in ld_path:
                env["LD_LIBRARY_PATH"] = f"{loader_dir}:{ld_path}" if ld_path else loader_dir

        # Force Vulkan device 0
        env["GGML_VK_VISIBLE_DEVICES"] = "0"
        return env

    def build_command(self) -> List[str]:
        """Builds llama-finetune CLI arguments adhering to pure GPU offload standards."""
        binary = self.resolve_binary_path()
        output_lora = os.path.join(self.config.output_dir, "adapter_lora.bin")

        cmd = [
            binary,
            "--model-base", self.config.model_path,
            "--train-data", self.config.dataset_path,
            "--lora-out", output_lora,
            "--lora-r", str(self.config.lora_r),
            "--lora-alpha", str(self.config.lora_alpha),
            "--batch-size", str(self.config.batch_size),
            "--grad-accum", str(self.config.gradient_accumulation_steps),
            "--epochs", str(self.config.epochs),
            "--threads", str(self.config.threads),
            "--ctx", str(self.config.context_length),
            "--save-every", str(self.config.save_every_steps),
        ]

        if self.config.strict_gpu:
            # Full GPU offload layers (99 offloads all transformer blocks)
            cmd.extend(["-ngl", "99"])

        return cmd

    def train(self) -> TrainingResult:
        """Executes llama-finetune and enforces zero silent fallback during backpropagation."""
        self.validate_environment()
        cmd = self.build_command()
        env = self.get_execution_environment()

        logger.info("[LlamaTrainer] Launching native training: %s", " ".join(cmd))
        start_time = time.time()
        last_loss = 0.0
        total_steps = 0

        # Pattern matches: loss = 1.2345, step 12/100
        loss_pattern = re.compile(r"(?:loss|cost)\s*=\s*([0-9\.]+)", re.IGNORECASE)
        step_pattern = re.compile(r"step\s*(\d+)", re.IGNORECASE)

        output_lora = os.path.join(self.config.output_dir, "adapter_lora.bin")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue

                logger.debug("[llama-finetune] %s", line)

                # Strict Check: Detect fallback or unsupported operators
                if self.config.strict_gpu:
                    if "fallback to cpu" in line.lower() or "unsupported op" in line.lower():
                        process.terminate()
                        raise GpuOperatorNotSupportedError(
                            operator_name="Backpropagation/AdamW GEMM",
                            backend="Vulkan",
                            unsupported_reasons=[line],
                        )

                # Track loss & steps
                loss_match = loss_pattern.search(line)
                if loss_match:
                    try:
                        last_loss = float(loss_match.group(1))
                    except ValueError:
                        pass

                step_match = step_pattern.search(line)
                if step_match:
                    try:
                        total_steps = int(step_match.group(1))
                    except ValueError:
                        pass

                # Hardware safety check
                self.monitor.check_safety_limits(stage=f"step_{total_steps}")

            ret_code = process.wait()
            if ret_code != 0:
                raise TermuxTrainError(
                    f"llama-finetune process exited with non-zero code {ret_code}",
                    error_code="NATIVE_PROCESS_FAILED",
                )

        except Exception as err:
            process.kill()
            raise err

        duration = time.time() - start_time
        return TrainingResult(
            success=True,
            total_steps=total_steps,
            final_loss=last_loss,
            duration_seconds=duration,
            output_artifact_path=output_lora,
            checkpoints=self.checkpoints,
            metadata={"backend": "vulkan", "model": self.config.model_path},
        )
