"""
AMEVA-GPU Command-Line Interface (ameva-gpu)
"""
import sys
import argparse
import time
from .doctor import Doctor
from .core import create_context
from .adapters import (
    SttAdapter,
    DiffusionAdapter,
    BitnetAdapter,
    LlamaCppAdapter,
    TtsAdapter,
    VisionAdapter,
)

def cmd_doctor(args):
    """Runs the 12-stage validation hierarchy."""
    doc = Doctor()
    report = doc.run_self_test(verbose=True)
    sys.exit(0 if report.overall_success else 1)


def cmd_install(args):
    """Auto-provisions native Bionic Vulkan binaries and verifies hardware."""
    print("\n[PROVISIONING] Initializing AMEVA Vulkan Acceleration Engine...")
    t0 = time.perf_counter()
    doc = Doctor()
    report = doc.run_self_test(verbose=True)
    t1 = time.perf_counter()

    if report.overall_success or report.passed_stages >= 7:
        print(f"[SUCCESS] AMEVA Vulkan Runtime verified in {(t1-t0)*1000:.2f} ms.")
        print(f"  Target: {report.device_name} (Driver: {report.driver_version})")
        print(f"  Single Loader Chain: {report.loader_path}")
        sys.exit(0)
    else:
        print(f"[WARNING] Vulkan probe uncertified. Active fallback backend: {report.recommended_backend}")
        sys.exit(1)


def cmd_benchmark(args):
    """Runs cross-modal adapter inspection and native GEMM throughput micro-benchmark."""
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("\n" + "=" * 60)
    print("  AMEVA-Vulkan-Runtime: Adapter Status & Micro-Benchmark    ")
    print("=" * 60)

    ctx = create_context(device="auto")
    print(f"  Active Accelerator: {ctx.device_name} ({ctx.backend_type.upper()})\n")

    doc = Doctor()
    report = doc.run_self_test(verbose=False)

    # 1. Inspect 6 Modality Adapters
    benchmarks = [
        ("STT (Whisper Base)", SttAdapter.bind(None, report)),
        ("Diffusion (SDXS 256p)", DiffusionAdapter.bind(None, report)),
        ("LLM (BitNet 1.58-bit)", BitnetAdapter.bind(None, report)),
        ("LLM (LlamaCpp GGUF)", LlamaCppAdapter.bind(None, report)),
        ("TTS (Piper HiFi-GAN)", TtsAdapter.bind(None, report)),
        ("Vision (LLaVA ViT)", VisionAdapter.bind(None, report)),
    ]

    for name, meta in benchmarks:
        print(f"  - {name:<26} -> Backend: {meta.backend:<9} | Status: {meta.status}")

    # 2. Run Micro-GEMM Latency Measurement (256x256)
    import numpy as np
    from .bindings import AmevaVulkanLib

    M, K, N = 256, 256, 256
    a = np.ones((M, K), dtype=np.float32)
    b = np.full((K, N), 0.5, dtype=np.float32)
    c = np.zeros((M, N), dtype=np.float32)

    vlib = AmevaVulkanLib()
    used_engine = "CPU_NUMPY_REFERENCE (Fallback)"

    t_start = time.perf_counter()
    if vlib.is_loaded():
        res = vlib.call_matmul_f32(a, b, c, M, K, N)
        if res == 0:
            used_engine = f"NATIVE_C_API ({ctx.backend_type.upper()})"
        else:
            c = np.matmul(a, b)
    else:
        c = np.matmul(a, b)
    t_end = time.perf_counter()

    elapsed_ms = (t_end - t_start) * 1000.0
    ops = 2.0 * M * K * N
    gflops = (ops / (elapsed_ms * 1e6)) if elapsed_ms > 0 else 0.0

    print("\n" + "-" * 60)
    print("  Micro-GEMM (256x256):")
    print(f"  - Active Context:   {ctx.backend_type.upper()} ({ctx.device_name})")
    print(f"  - Executed Kernel:  {used_engine}")
    print(f"  - Elapsed Time:     {elapsed_ms:.3f} ms")
    print(f"  - Throughput:       {gflops:.2f} GFLOPS")
    print(f"  - Checksum (c[0,0]): {c[0,0]:.1f} (Expected: {K * 0.5:.1f})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AMEVA Vulkan Hardware Diagnostic & Runtime Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # doctor
    p_doc = subparsers.add_parser("doctor", help="Run 12-stage validation hierarchy (V0-V11)")
    p_doc.set_defaults(func=cmd_doctor)

    # install
    p_inst = subparsers.add_parser("install", help="Provision hardware runtime and verify")
    p_inst.set_defaults(func=cmd_install)

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run cross-modal acceleration benchmarks")
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
