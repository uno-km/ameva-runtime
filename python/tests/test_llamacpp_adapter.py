"""
Unit tests for LlamaCppAdapter adhering to Zero-Silent-Fallback and Fail-Fast standards.
"""
import unittest

from ameva_runtime.vulkan.doctor import DiagnosticReport
from ameva_runtime.vulkan.protocol import BindingResult
from ameva_runtime.vulkan.adapters.llamacpp import (
    LlamaCppAdapter,
    verify_vulkan_llm_output,
    _calculate_llama_layers,
    DEFAULT_GPU_LAYERS,
)
from ameva_runtime.exceptions import AmevaRuntimeError, PlatformNotSupportedError


class TestLlamaCppAdapterRefactored(unittest.TestCase):
    def setUp(self):
        self.vulkan_report = DiagnosticReport(
            overall_success=True,
            device_name="Mali-G78",
            driver_version="1.3.219",
            loader_path="/system/lib64/libvulkan.so",
            vendor_id=0x13B5,
            passed_stages=11,
            total_stages=11,
            total_elapsed_ms=4.5,
            recommended_backend="vulkan",
        )
        self.cpu_report = DiagnosticReport(
            overall_success=False,
            device_name="Cortex-A78",
            driver_version="",
            loader_path="",
            vendor_id=0x0,
            passed_stages=0,
            total_stages=11,
            total_elapsed_ms=0.1,
            recommended_backend="cpu_neon",
        )

    def test_p0_1_vulkan_skip_checks_not_injected(self):
        """P0-1 Fix: GGML_VULKAN_SKIP_CHECKS must never be injected by default."""
        env = LlamaCppAdapter.get_execution_environment()
        self.assertNotIn("GGML_VULKAN_SKIP_CHECKS", env)

    def test_p0_2_unsupported_backend_raises_error(self):
        """P0-2 Fix: Unsupported backend string must raise AmevaRuntimeError."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(report=self.vulkan_report, requested_backend="metal")
        self.assertIn("Unsupported backend", str(ctx.exception))

    def test_p0_2_explicit_vulkan_without_hardware_raises_platform_error(self):
        """P0-2 Fix: Explicit vulkan request without hardware MUST fail-fast without silent fallback."""
        with self.assertRaises(PlatformNotSupportedError):
            LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="vulkan")

        with self.assertRaises(PlatformNotSupportedError):
            LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="gpu")

    def test_p0_2_auto_backend_adaptive_routing_on_cpu(self):
        """P0-2: auto routes adaptively to cpu_neon when Vulkan is unavailable."""
        res = LlamaCppAdapter.bind(report=self.cpu_report, requested_backend="auto")
        self.assertEqual(res.backend, "cpu_neon")
        self.assertIn("CPU", res.status)

    def test_p0_3_conflicting_ngl_raises_error(self):
        """P0-3 / P0-6 Fix: Requesting Vulkan with requested_ngl <= 0 must fail fast."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                report=self.vulkan_report,
                requested_backend="vulkan",
                requested_ngl=0,
            )
        self.assertIn("Conflicting configuration", str(ctx.exception))

        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                report=self.vulkan_report,
                requested_backend="vulkan",
                requested_ngl=-5,
            )
        self.assertIn("Conflicting configuration", str(ctx.exception))

    def test_p0_4_atomic_mutation_and_precedence(self):
        """P0-4 Fix: Dict engine mutation must set values deterministically."""
        engine_dict = {"ngl": 0, "device": "none"}
        res = LlamaCppAdapter.bind(engine=engine_dict, report=self.vulkan_report, requested_ngl=28)
        self.assertEqual(res.status, "CONFIGURED_VULKAN")
        self.assertEqual(engine_dict["ngl"], 28)
        self.assertEqual(engine_dict["device"], "vulkan")

    def test_p1_4_status_is_configured_vulkan(self):
        """P1-4 Fix: Status must accurately reflect CONFIGURED_VULKAN."""
        res = LlamaCppAdapter.bind(report=self.vulkan_report)
        self.assertEqual(res.status, "CONFIGURED_VULKAN")
        self.assertTrue(res.is_vulkan)

    def test_ld_library_path_not_injected(self):
        """Fix: engine dict must NOT receive arbitrary LD_LIBRARY_PATH injection."""
        engine_dict = {}
        LlamaCppAdapter.bind(engine=engine_dict, report=self.vulkan_report)
        self.assertNotIn("env", engine_dict)

    def test_cli_list_vulkan_conflicting_ngl_raises_error(self):
        """Fix: Vulkan requested with existing CLI list -ngl 0 must raise AmevaRuntimeError."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                engine=["-ngl", "0"],
                report=self.vulkan_report,
                requested_backend="vulkan",
            )
        self.assertIn("Conflicting CLI arguments", str(ctx.exception))

    def test_cli_list_cpu_conflicting_ngl_raises_error(self):
        """Fix: CPU requested with existing CLI list -ngl 32 must raise AmevaRuntimeError."""
        with self.assertRaises(AmevaRuntimeError) as ctx:
            LlamaCppAdapter.bind(
                engine=["-ngl", "32"],
                report=self.vulkan_report,
                requested_backend="cpu",
            )
        self.assertIn("Conflicting CLI arguments", str(ctx.exception))

    def test_cli_list_clean_injection(self):
        """Fix: Clean CLI list receives -ngl 999 and --device vulkan without duplicate."""
        cmd = ["-m", "model.gguf"]
        LlamaCppAdapter.bind(engine=cmd, report=self.vulkan_report, requested_backend="vulkan")
        self.assertIn("-ngl", cmd)
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")
        self.assertIn("--device", cmd)
        self.assertEqual(cmd[cmd.index("--device") + 1], "vulkan")

    def test_p1_7_verify_vulkan_llm_output_with_returncode(self):
        """P1-7 Fix: verify_vulkan_llm_output must inspect returncode and stdout/stderr."""
        with self.assertRaises(AmevaRuntimeError):
            verify_vulkan_llm_output("any output", returncode=139)

        with self.assertRaises(PlatformNotSupportedError):
            verify_vulkan_llm_output("llama_model_load: falling back to cpu", returncode=0)

        with self.assertRaises(PlatformNotSupportedError):
            verify_vulkan_llm_output("Normal text without positive vulkan init", returncode=0)

        # Positive output
        verify_vulkan_llm_output("ggml_vulkan: using device Mali-G78\nGeneration successful", returncode=0)

    def test_build_cli_args_vulkan_and_fail_fast(self):
        """CLI argument assembly validation."""
        args = LlamaCppAdapter.build_cli_args(
            executable="llama-cli",
            model_path="/path/to/model.gguf",
            target_backend="vulkan",
            ngl_override=32,
            device_name="Mali-G78",
        )
        self.assertIn("-ngl", args)
        self.assertEqual(args[args.index("-ngl") + 1], "32")
        self.assertIn("--device", args)
        self.assertEqual(args[args.index("--device") + 1], "Mali-G78")

        with self.assertRaises(AmevaRuntimeError):
            LlamaCppAdapter.build_cli_args(
                executable="llama-cli",
                model_path="/path/to/model.gguf",
                target_backend="vulkan",
                ngl_override=0,
            )

        with self.assertRaises(AmevaRuntimeError):
            LlamaCppAdapter.build_cli_args(
                executable="llama-cli",
                model_path="/path/to/model.gguf",
                target_backend="invalid_backend",
            )

    def test_resolve_binary_path_missing_raises_e001(self):
        """Zero-Silent-Fallback: Missing binary strictly raises AmevaLlamaAssetMissingError (AMEVA-LLAMA-E001)."""
        import tempfile
        from unittest.mock import patch
        from ameva_runtime.exceptions import AmevaLlamaAssetMissingError
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"HOME": tmpdir}):
                with self.assertRaises(AmevaLlamaAssetMissingError) as ctx:
                    LlamaCppAdapter.resolve_binary_path()
                self.assertEqual(ctx.exception.error_code, "AMEVA-LLAMA-E001")

    def test_resolve_binary_path_manifest_missing_raises_e002(self):
        """Integrity Guard: Binary present without cryptographic manifest raises AmevaLlamaVerificationError (AMEVA-LLAMA-E002)."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from ameva_runtime.exceptions import AmevaLlamaVerificationError
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "llama-cli"
            bin_file.write_bytes(b"ELF_FAKE_BIN")
            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                with self.assertRaises(AmevaLlamaVerificationError) as ctx:
                    LlamaCppAdapter.resolve_binary_path()
                self.assertEqual(ctx.exception.error_code, "AMEVA-LLAMA-E002")

    def test_resolve_binary_path_hash_mismatch_raises_e002(self):
        """Integrity Guard: Cryptographic hash mismatch raises AmevaLlamaVerificationError (AMEVA-LLAMA-E002)."""
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from ameva_runtime.exceptions import AmevaLlamaVerificationError
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "llama-cli"
            bin_file.write_bytes(b"ELF_FAKE_BIN")

            manifest_dir = Path(tmpdir) / ".local" / "share" / "ameva" / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / "llamacpp.json"
            manifest_file.write_text(json.dumps({
                "bundle_id": "llamacpp",
                "binary_sha256": "0" * 64,
            }), encoding="utf-8")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                with self.assertRaises(AmevaLlamaVerificationError) as ctx:
                    LlamaCppAdapter.resolve_binary_path()
                self.assertEqual(ctx.exception.error_code, "AMEVA-LLAMA-E002")

    def test_resolve_binary_path_success(self):
        """Authentic asset resolution: Returns resolved binary path when verified."""
        import hashlib
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "llama-cli"
            content = b"ELF_VALID_TEST_BINARY"
            bin_file.write_bytes(content)
            bin_sha = hashlib.sha256(content).hexdigest()

            manifest_dir = Path(tmpdir) / ".local" / "share" / "ameva" / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / "llamacpp.json"
            manifest_file.write_text(json.dumps({
                "bundle_id": "llamacpp",
                "binary_sha256": bin_sha,
            }), encoding="utf-8")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                resolved = LlamaCppAdapter.resolve_binary_path()
                self.assertEqual(resolved, str(bin_file.resolve()))


if __name__ == "__main__":
    unittest.main()