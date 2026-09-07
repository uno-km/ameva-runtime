"""
AMEVA Runtime Unified Command-Line Interface (CLI)
=================================================
Commands:
  - doctor    : Comprehensive 12-stage hardware & driver diagnostic.
  - profile   : Show detected SoC, GPU, CPU cgroup affinity, and recommendation.
  - plan      : Dry-run the smart router execution plan for a model.
  - exec      : Execute model inference through the optimal/safe backend.
  - benchmark : Inspect multi-modal adapters and run GEMM micro-benchmark.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from typing import List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from .core import AmevaRuntime, get_runtime
from .detector import detect_hardware, HardwareProfile
from .doctor import Doctor
from .router import SmartRouter
from .exceptions import AmevaRuntimeError

logger = logging.getLogger("ameva_runtime.cli")


def cmd_doctor(args: argparse.Namespace) -> int:
    """Runs the 12-stage diagnostic self-test."""
    doc = Doctor()
    report = doc.run_self_test(verbose=True)
    if report.overall_success or report.passed_stages >= 7:
        print("\n[RESULT] Diagnostic completed successfully.")
        return 0
    else:
        print("\n[RESULT] Diagnostic flagged issues in critical stages.")
        return 1


def cmd_profile(args: argparse.Namespace) -> int:
    """Displays detailed hardware, GPU, SoC, and CPU cgroup affinity information."""
    runtime = get_runtime()
    prof = runtime.profile

    print("=" * 65)
    print("  AMEVA Runtime: Hardware & System Topology Profile")
    print("=" * 65)
    print(f"  Vendor / Architecture : {prof.vendor} ({prof.arch})")
    print(f"  SoC Model             : {prof.soc_model}")
    print(f"  GPU Family / Driver   : {prof.gpu_family} (Driver: {prof.driver_version})")
    print(f"  Vulkan Loader Available: {'YES' if prof.has_vulkan_loader else 'NO'}")
    print(f"  OpenCL Available      : {'YES' if prof.has_opencl else 'NO'}")
    print(f"  NPU Available         : {'YES' if prof.has_npu else 'NO'}")
    print("-" * 65)
    print(f"  CPU Online Cores      : {prof.cpu_cores}")
    print(f"  CPU Allowed Cores     : {prof.allowed_cpus} (Length: {len(prof.allowed_cpus)})")
    print(f"  Cgroup Restrained     : {'YES ( samsung /moderate cgroup active )' if prof.is_cgroup_restrained else 'NO'}")
    print(f"  Memory Available      : {prof.total_ram_mb} MB total ({prof.available_ram_mb} MB free)")
    print("-" * 65)
    print(f"  Recommended Backend   : {prof.recommended_backend.upper()}")
    print(f"  Optimal Thread Count  : {prof.recommended_threads}")
    print("=" * 65)
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    """Shows the execution plan determined by the SmartRouter."""
    runtime = get_runtime()
    plan = runtime.plan_execution(
        model_name=args.model or "",
        requested_backend=args.backend
    )

    print("=" * 65)
    print("  AMEVA Runtime: Smart Router Execution Plan")
    print("=" * 65)
    print(f"  Model Target       : {args.model or '(general)'}")
    print(f"  Selected Backend   : {plan.backend.upper()}")
    print(f"  GPU Layer Offload  : {plan.ngl}")
    print(f"  Worker Threads     : {plan.threads}")
    print(f"  Pin CPU Cores      : {plan.affinity_cpus}")
    print(f"  Batch Size         : {plan.batch_size}")
    print(f"  Context Window     : {plan.context_size}")
    print(f"  Safety Rationale   : {plan.rationale}")
    print("=" * 65)
    return 0


def cmd_exec(args: argparse.Namespace) -> int:
    """Executes model inference safely according to the SmartRouter plan."""
    runtime = get_runtime()
    plan = runtime.plan_execution(
        model_name=args.model,
        requested_backend=args.backend
    )

    print(f"[AMEVA-RUN] Route selected: {plan.backend.upper()} (NGL={plan.ngl}, Threads={plan.threads})")
    print(f"[AMEVA-RUN] Rationale: {plan.rationale}\n")

    try:
        result = runtime.execute(
            model_path=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            backend=args.backend,
        )
        print(f"[EXEC] Command: {' '.join(result.command)}\n")
        print("=== GENERATED OUTPUT ===")
        print(result.text)
        print("=" * 65)
        print(f"  Backend Used     : {result.backend_used}")
        if result.tokens_per_second > 0:
            print(f"  Token Generation : {result.eval_tokens} tokens ({result.tokens_per_second:.2f} t/s)")
        if result.prompt_tokens_per_second > 0:
            print(f"  Prompt Eval      : {result.prompt_tokens} tokens ({result.prompt_tokens_per_second:.2f} t/s)")
        print(f"  Total Latency    : {result.total_time_ms:.1f} ms")
        print("=" * 65)
        return result.return_code
    except AmevaRuntimeError as e:
        print(f"[ERROR] {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Inference execution failed: {e}")
        return 1



def cmd_benchmark(args: argparse.Namespace) -> int:
    """Runs adapter inspection and GEMM micro-benchmarks."""
    from .adapters import (
        SttAdapter,
        DiffusionAdapter,
        BitnetAdapter,
        LlamaCppAdapter,
        TtsAdapter,
        VisionAdapter,
    )
    import numpy as np

    print("=" * 65)
    print("  AMEVA Runtime: Multi-Modal Adapter Status & Benchmark")
    print("=" * 65)

    runtime = get_runtime()
    prof = runtime.profile

    adapters = [
        ("STT (Whisper)", SttAdapter.bind(None, prof)),
        ("Diffusion (SDXS)", DiffusionAdapter.bind(None, prof)),
        ("LLM (BitNet)", BitnetAdapter.bind(None, prof)),
        ("LLM (LlamaCpp)", LlamaCppAdapter.bind(None, prof)),
        ("TTS (Piper)", TtsAdapter.bind(None, prof)),
        ("Vision (ViT)", VisionAdapter.bind(None, prof)),
    ]

    for name, binding in adapters:
        status = "READY" if binding.is_accelerated or binding.device_id != -1 else "FALLBACK"
        print(f"  - {name:<22} -> Backend: {binding.backend.upper():<8} | Status: {status}")

    print("-" * 65)
    print("  Running Micro-GEMM Benchmark (256x256 Float32)...")
    M, K, N = 256, 256, 256
    a = np.ones((M, K), dtype=np.float32)
    b = np.full((K, N), 0.5, dtype=np.float32)

    t0 = time.perf_counter()
    c = np.matmul(a, b)
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    gflops = (2.0 * M * K * N) / (elapsed_ms * 1e6) if elapsed_ms > 0 else 0.0

    print(f"  Matrix Multiplication (256x256) : {elapsed_ms:.3f} ms ({gflops:.2f} GFLOPS)")
    print("=" * 65)
    return 0


def _find_llama_cli() -> Optional[str]:
    search_paths = [
        os.path.expanduser("~/vulkan-llama/bin/llama-cli"),
        "/data/data/com.termux/files/home/vulkan-llama/bin/llama-cli",
        os.path.expanduser("~/.termux-llama/bin/llama-cli"),
        "/data/data/com.termux/files/home/.termux-llama/bin/llama-cli",
        os.path.expanduser("~/BitNet_ms/3rdparty/llama.cpp/build-vulkan/bin/llama-cli"),
        "/data/data/com.termux/files/home/BitNet_ms/3rdparty/llama.cpp/build-vulkan/bin/llama-cli",
        "/data/data/com.termux/files/usr/bin/llama-cli",
        "llama-cli",
    ]
    for p in search_paths:
        if os.path.isabs(p) and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
        # Check PATH
        import shutil
        found = shutil.which(p)
        if found:
            return found
    return None


def _resolve_model_path(model_arg: str) -> str:
    if os.path.exists(model_arg):
        return model_arg
    # Common Termux locations
    candidates = [
        os.path.expanduser(f"~/.termux-llama/models/{model_arg}"),
        os.path.expanduser(f"~/.termux-llama/models/{model_arg}.gguf"),
        f"/data/data/com.termux/files/home/.termux-llama/models/{model_arg}",
        f"/data/data/com.termux/files/home/.termux-llama/models/{model_arg}.gguf",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return model_arg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ameva-run",
        description="AMEVA Unified Next-Gen On-Device AI Runtime CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # doctor
    subparsers.add_parser("doctor", help="Run 12-stage hardware diagnostic")

    # profile
    subparsers.add_parser("profile", help="Show system and hardware topology profile")

    # plan
    plan_parser = subparsers.add_parser("plan", help="Dry-run execution plan")
    plan_parser.add_argument("-m", "--model", default="", help="Model name or path")
    plan_parser.add_argument("-b", "--backend", default=None, choices=["auto", "vulkan", "opencl", "cpu", "npu"])

    # exec
    exec_parser = subparsers.add_parser("exec", help="Safely execute inference with optimal backend")
    exec_parser.add_argument("-m", "--model", required=True, help="Model name or path (.gguf)")
    exec_parser.add_argument("-p", "--prompt", default="Hello! Who are you?", help="Input prompt")
    exec_parser.add_argument("-n", "--max-tokens", type=int, default=64, help="Max tokens to generate")
    exec_parser.add_argument("-b", "--backend", default=None, choices=["auto", "vulkan", "opencl", "cpu", "npu"])
    exec_parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")

    # benchmark
    subparsers.add_parser("benchmark", help="Inspect adapters and benchmark GEMM")

    return parser


def main() -> int:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()
    if args.command == "doctor":
        return cmd_doctor(args)
    elif args.command == "profile":
        return cmd_profile(args)
    elif args.command == "plan":
        return cmd_plan(args)
    elif args.command == "exec":
        return cmd_exec(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        parser.print_help()
        return 0


# Backward compatibility entry point for 'ameva-gpu'
def legacy_gpu_main() -> int:
    # If called as legacy ameva-gpu, map arguments or delegate
    parser = argparse.ArgumentParser(
        prog="ameva-gpu",
        description="AMEVA Unified GPU/Hardware Runtime (Legacy Compatible Entry Point)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Legacy Subcommands")
    subparsers.add_parser("doctor", help="Run diagnostic")
    subparsers.add_parser("install", help="Inspect and configure hardware profile")
    subparsers.add_parser("benchmark", help="Benchmark adapters")

    args, unknown = parser.parse_known_args()
    if args.command == "doctor" or args.command == "install":
        return cmd_doctor(args)
    elif args.command == "benchmark":
        return cmd_benchmark(args)
    else:
        # Fallback to main CLI parser
        return main()


if __name__ == "__main__":
    sys.exit(main())
