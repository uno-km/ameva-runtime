"""
E2E verification script for ameva-runtime on real device (Galaxy S21 / S22).
"""
import sys
import ameva_runtime
from ameva_runtime.core import resolve_model_path, plan
from ameva_runtime.exceptions import ModelNotFoundError, AmbiguousModelMatchError

print("=== [1] Version Check ===")
print(f"ameva_runtime version: {ameva_runtime.__version__}")

print("\n=== [2] Model Resolution: Nonexistent Model ===")
try:
    resolve_model_path("non_existent_model_xyz")
    print("FAIL: Expected ModelNotFoundError")
    sys.exit(1)
except ModelNotFoundError as e:
    print(f"PASS: Caught {e.__class__.__name__}: {e.error_code}")

print("\n=== [3] Model Resolution: Ambiguous Match Detection ===")
try:
    resolve_model_path("qwen2.5-0.5b")
    print("FAIL: Expected AmbiguousModelMatchError")
    sys.exit(1)
except AmbiguousModelMatchError as e:
    print(f"PASS: Caught {e.__class__.__name__}: {e.error_code}")
    print(f"      Candidates detected: {len(e.candidates)} models -> {e.candidates}")

print("\n=== [4] Model Resolution: Disambiguated Single Match ===")
resolved = resolve_model_path("qwen2.5-0.5b-instruct-q4")
print(f"PASS: Resolved exactly to: {resolved}")
assert "q4_k_m" in resolved

print("\n=== [4] SmartRouter Plan: CPU Mode ===")
p_cpu = plan(backend="cpu", ngl=0)
print(f"Backend: {p_cpu.backend}, Threads: {p_cpu.threads}, NGL: {p_cpu.ngl}")
assert p_cpu.backend == "cpu_neon"
assert p_cpu.ngl == 0

print("\n=== [5] SmartRouter Plan: Vulkan Mode with Custom NGL ===")
p_vk = plan(backend="vulkan", ngl=17)
print(f"Backend: {p_vk.backend}, Threads: {p_vk.threads}, NGL: {p_vk.ngl}")
assert p_vk.backend == "vulkan"
assert p_vk.ngl == 17

print("\n=== [6] Full On-Device Inference via ameva_runtime.run() ===")
res = ameva_runtime.run(
    model="qwen2.5-0.5b-instruct-q4",
    prompt="Hello!",
    max_tokens=16,
    backend="cpu",
)
print(f"PASS: Inference completed successfully!")
print(f"      Model: {res.model_name}")
print(f"      Backend used: {res.backend_used}")
print(f"      TPS: {res.tokens_per_second:.1f} t/s")
print(f"      Output: {res.text.strip()}")

print("\n=== ALL E2E INVARIANTS VERIFIED SUCCESSFULLY ===")
