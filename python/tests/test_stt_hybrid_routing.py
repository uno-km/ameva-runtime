"""
Unit Tests: STT (Whisper) Hardware-Aware Adaptive Routing & Quirks
===================================================================
Verifies that SmartRouter and SttAdapter produce optimal Vulkan plans
for Galaxy S25 (Adreno 830) and Galaxy A35 (Mali-G68) with correct
Medium MatMul quirks, library paths, and clean CPU fallbacks.
"""
import os
import unittest
from ameva_runtime.detector import HardwareProfile
from ameva_runtime.router import SmartRouter
from ameva_runtime.adapters.stt import SttAdapter


class TestSttHybridRouting(unittest.TestCase):
    def setUp(self):
        self.a35_profile = HardwareProfile(
            vendor="samsung",
            soc_model="s5e8835",
            gpu_family="mali",
            has_kgsl_node=False,
            has_mali_node=True,
            total_cpu_cores=8,
            allowed_cpu_set={0, 1, 2, 3, 4, 5, 6, 7},
            big_core_indices=[4, 5, 6, 7],
            little_core_indices=[0, 1, 2, 3],
            recommended_threads=4,
            recommended_backend="vulkan",
            hardware_hazard=None,
            diagnosis_reason="Galaxy A35 Exynos 1380 Mali-G68",
        )

        self.s25_profile = HardwareProfile(
            vendor="qualcomm",
            soc_model="sm8750",
            gpu_family="adreno",
            has_kgsl_node=True,
            has_mali_node=False,
            total_cpu_cores=8,
            allowed_cpu_set={0, 1, 2, 3, 4, 5, 6, 7},
            big_core_indices=[2, 3, 4, 5, 6, 7],
            little_core_indices=[0, 1],
            recommended_threads=4,
            recommended_backend="vulkan",
            hardware_hazard=None,
            diagnosis_reason="Galaxy S25 Snapdragon 8 Elite Adreno 830",
        )

        self.cpu_only_profile = HardwareProfile(
            vendor="generic",
            soc_model="unknown",
            gpu_family="unknown",
            has_kgsl_node=False,
            has_mali_node=False,
            total_cpu_cores=4,
            allowed_cpu_set={0, 1, 2, 3},
            big_core_indices=[2, 3],
            little_core_indices=[0, 1],
            recommended_threads=2,
            recommended_backend="cpu_neon",
            hardware_hazard=None,
            diagnosis_reason="Generic CPU",
        )

    def test_a35_stt_vulkan_routing_and_mali_quirks(self):
        """Galaxy A35 (Mali-G68) must route to Vulkan with Medium MatMul and F16 disable quirks."""
        router = SmartRouter(self.a35_profile)
        plan = router.route_for_stt(model_name_or_path="ggml-large-v3-turbo-q5_0.bin")

        self.assertEqual(plan.backend, "vulkan")
        self.assertTrue(plan.is_gpu_accelerated)
        self.assertIn("-dev", plan.cli_flags)
        self.assertIn("0", plan.cli_flags)
        self.assertEqual(plan.env_overrides.get("GGML_VK_FORCE_MEDIUM_MATMUL"), "1")
        self.assertEqual(plan.env_overrides.get("GGML_VK_DISABLE_F16"), "1")
        self.assertIn("Medium MatMul", plan.diagnosis)

    def test_s25_stt_vulkan_routing_without_mali_quirks(self):
        """Galaxy S25 (Adreno 830) must route to Vulkan without Mali-specific quirks."""
        router = SmartRouter(self.s25_profile)
        plan = router.route_for_stt(model_name_or_path="ggml-large-v3-turbo-q5_0.bin")

        self.assertEqual(plan.backend, "vulkan")
        self.assertTrue(plan.is_gpu_accelerated)
        self.assertIn("-dev", plan.cli_flags)
        self.assertIn("0", plan.cli_flags)
        self.assertNotIn("GGML_VK_FORCE_MEDIUM_MATMUL", plan.env_overrides)
        self.assertNotIn("GGML_VK_DISABLE_F16", plan.env_overrides)

    def test_stt_explicit_cpu_routing(self):
        """When CPU is explicitly requested, router must produce clean CPU NEON plan."""
        router = SmartRouter(self.a35_profile)
        plan = router.route_for_stt(requested_backend="cpu_neon")

        self.assertEqual(plan.backend, "cpu_neon")
        self.assertFalse(plan.is_gpu_accelerated)
        self.assertIn("-dev", plan.cli_flags)
        self.assertIn("-1", plan.cli_flags)

    @staticmethod
    def _make_arm64_elf(hint: bytes = b"libvulkan.so") -> bytes:
        header = b"\x7fELF\x02\x01\x01" + (b"\x00" * 9) + b"\x02\x00" + (183).to_bytes(2, "little")
        header += b"\x00" * (64 - len(header))
        return header + (b"\x00" * 128) + hint + (b"\x00" * 128)

    def test_stt_adapter_bind_vulkan_with_verified_binary(self):
        """SttAdapter.bind should produce VULKAN_CANDIDATE_SELECTED status and config on Vulkan device."""
        import tempfile
        import json
        import hashlib
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper-cli"
            vk_content = self._make_arm64_elf(b"libvulkan.so")
            vk_bin.write_bytes(vk_content)
            bin_sha = hashlib.sha256(vk_content).hexdigest()

            manifest_p = Path(td) / "manifest.json"
            manifest_p.write_text(json.dumps({
                "bundle_id": "stt",
                "backend_feature": "vulkan",
                "target_architecture": "aarch64",
                "binary_sha256": bin_sha,
                "deployed_files": {},
            }), encoding="utf-8")

            class MockEngine:
                def __init__(self):
                    self.device = None
                    self.threads = None
                    self.binary_path = str(vk_bin)
                    self.manifest_path = str(manifest_p)

            engine = MockEngine()
            res = SttAdapter.bind(engine=engine, profile=self.a35_profile)

            self.assertEqual(res.status, "VULKAN_CANDIDATE_SELECTED")
            self.assertEqual(res.backend, "vulkan")
            self.assertFalse(res.is_vulkan)  # False until live GPU inference probe is verified
            self.assertEqual(engine.device, "vulkan_candidate")
            self.assertEqual(engine.threads, 4)
            self.assertEqual(engine.binary_path, str(vk_bin.resolve()))
            self.assertIn("env_overrides", res.config)
            self.assertEqual(res.config["env_overrides"].get("GGML_VK_FORCE_MEDIUM_MATMUL"), "1")
            self.assertEqual(res.config["env_overrides"].get("GGML_VK_DISABLE_F16"), "1")
            self.assertFalse(res.config["encoder_fp16"])  # strictly false when disable_f16 is true

    def test_has_vulkan_link_hint_detection(self):
        """Verifies that has_vulkan_link_hint correctly identifies Vulkan ELF binaries and rejects CPU binaries."""
        import tempfile
        from pathlib import Path
        from ameva_runtime.adapters.stt import has_vulkan_link_hint

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper_vk"
            cpu_bin = Path(td) / "whisper_cpu"
            txt_file = Path(td) / "not_elf.txt"

            vk_bin.write_bytes(self._make_arm64_elf(b"libggml-vulkan.so"))
            cpu_bin.write_bytes(self._make_arm64_elf(b"libggml-cpu.so"))
            txt_file.write_text("Hello Vulkan")

            self.assertTrue(has_vulkan_link_hint(vk_bin))
            self.assertFalse(has_vulkan_link_hint(cpu_bin))
            self.assertFalse(has_vulkan_link_hint(txt_file))

    def test_cli_compatibility_probe_honest_status(self):
        """Directive 1, 2, 4, 5: CLI probe must not claim Vulkan verification, device name, or false fallback."""
        import tempfile
        from pathlib import Path
        from ameva_runtime.adapters.stt import run_cli_compatibility_probe

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper-cli"
            vk_bin.write_bytes(self._make_arm64_elf(b"libvulkan.so"))

            probe = run_cli_compatibility_probe(vk_bin, {})
            self.assertEqual(probe.status, "CLI_COMPATIBLE")
            self.assertTrue(probe.is_cli_compatible)
            self.assertEqual(probe.device_name, "unknown")
            self.assertEqual(probe.backend_name, "unverified")
            self.assertIsNone(probe.cpu_fallback_detected)

    def test_stt_adapter_fail_fast_when_cpu_only_binary_provided(self):
        """Strict Zero-Silent-Fallback: Providing CPU-only binary when Vulkan required must raise AmevaRuntimeError."""
        import tempfile
        from pathlib import Path
        from ameva_runtime.exceptions import AmevaRuntimeError

        with tempfile.TemporaryDirectory() as td:
            cpu_bin = Path(td) / "whisper-cli"
            cpu_bin.write_bytes(self._make_arm64_elf(b"libggml-cpu.so"))

            class MockEngine:
                def __init__(self):
                    self.device = None
                    self.threads = None
                    self.binary_path = str(cpu_bin)

            engine = MockEngine()
            with self.assertRaises(AmevaRuntimeError):
                SttAdapter.bind(
                    engine=engine,
                    profile=self.a35_profile,
                )

    def test_single_explicit_candidate_isolation(self):
        """Directive 6: If explicit candidate is missing or invalid, fail immediately without trying canonical."""
        from pathlib import Path
        from ameva_runtime.exceptions import AmevaRuntimeError

        class MockEngine:
            def __init__(self):
                self.binary_path = "/non/existent/whisper-cli"

        engine = MockEngine()
        with self.assertRaises(AmevaRuntimeError) as ctx:
            SttAdapter.bind(engine=engine, profile=self.a35_profile)
        self.assertIn("Explicit STT binary does not exist", str(ctx.exception))

    def test_transactional_mutation_and_exact_unbind_restoration(self):
        """Directive 10 & 11: Transactional rollback, candidate phase isolation, and faithful snapshot restoration on unbind."""
        import tempfile
        import json
        import hashlib
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper-cli"
            vk_content = self._make_arm64_elf(b"libvulkan.so")
            vk_bin.write_bytes(vk_content)
            bin_sha = hashlib.sha256(vk_content).hexdigest()

            manifest_p = Path(td) / "manifest.json"
            manifest_p.write_text(json.dumps({
                "bundle_id": "stt",
                "backend_feature": "vulkan",
                "target_architecture": "aarch64",
                "binary_sha256": bin_sha,
                "deployed_files": {},
            }), encoding="utf-8")

            class MockConfig:
                def __init__(self):
                    self.extra = {}

            class MockEngine:
                def __init__(self):
                    self.device = "auto"
                    self.threads = 8
                    self.binary_path = str(vk_bin)
                    self.manifest_path = str(manifest_p)
                    self.config = MockConfig()
                    self.custom_attr = "preserve_me"

            engine = MockEngine()
            binding = SttAdapter.bind(engine=engine, profile=self.a35_profile)

            # P0-4: Candidate phase must NOT claim verified GPU or call set_vulkan(True)
            self.assertEqual(binding.status, "VULKAN_CANDIDATE_SELECTED")
            self.assertFalse(binding.is_vulkan)
            self.assertEqual(engine.config.extra["requested_backend"], "vulkan")
            self.assertFalse(engine.config.extra["backend_verified"])
            self.assertNotIn("use_vulkan", engine.config.extra)
            self.assertEqual(engine.device, "vulkan_candidate")
            self.assertEqual(engine.threads, 4)

            # Unbind should restore EXACT pre-binding state, NOT force cpu
            SttAdapter.unbind(engine=engine, binding_result=binding)
            self.assertEqual(engine.device, "auto")
            self.assertEqual(engine.threads, 8)
            self.assertEqual(engine.custom_attr, "preserve_me")
            self.assertNotIn("requested_backend", engine.config.extra)

    def test_explicit_binary_strictly_requires_manifest(self):
        """P0-3: Production bind strictly requires manifest; engine cannot bypass verification."""
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from ameva_runtime.adapters.stt import resolve_unmanaged_whisper_binary_for_development
        from ameva_runtime.exceptions import AmevaRuntimeError

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper-cli"
            vk_content = self._make_arm64_elf(b"libvulkan.so")
            vk_bin.write_bytes(vk_content)
            bin_sha = hashlib.sha256(vk_content).hexdigest()

            class MockEngine:
                def __init__(self):
                    self.binary_path = str(vk_bin)

            engine = MockEngine()
            # 1. Production bind without manifest must strictly raise AmevaRuntimeError
            with self.assertRaises(AmevaRuntimeError) as ctx:
                SttAdapter.bind(engine=engine, profile=self.a35_profile)
            self.assertIn("STT deployment manifest is missing", str(ctx.exception))

            # 2. Developer diagnostic function allows resolution only when called explicitly
            dev_bin = resolve_unmanaged_whisper_binary_for_development(engine)
            self.assertEqual(dev_bin, vk_bin.resolve())

            # 3. With explicit manifest: production bind succeeds
            manifest_p = Path(td) / "manifest.json"
            manifest_p.write_text(json.dumps({
                "bundle_id": "stt",
                "backend_feature": "vulkan",
                "target_architecture": "aarch64",
                "binary_sha256": bin_sha,
                "deployed_files": {},
            }), encoding="utf-8")
            binding = SttAdapter.bind(engine=engine, profile=self.a35_profile, manifest_path=manifest_p)
            self.assertEqual(binding.backend, "vulkan")

    def test_quirk_table_mali_g78_g68_vs_adreno(self):
        """Directive 8: Only verified GPUs in quirk table disable FP16."""
        from ameva_runtime.adapters.stt import resolve_stt_quirks
        from types import SimpleNamespace

        # Mali-G78 (Exynos 2100)
        report_g78 = SimpleNamespace(vendor_id=0x13B5, device_name="Mali-G78")
        q_g78 = resolve_stt_quirks(report_g78)
        self.assertTrue(q_g78.disable_fp16)
        self.assertTrue(q_g78.force_medium_matmul)

        # Mali-G68 (Exynos 1380)
        report_g68 = SimpleNamespace(vendor_id=0x13B5, device_name="Mali-G68")
        q_g68 = resolve_stt_quirks(report_g68)
        self.assertTrue(q_g68.disable_fp16)
        self.assertTrue(q_g68.force_medium_matmul)

        # Adreno 830 (Snapdragon 8 Elite)
        report_adreno = SimpleNamespace(vendor_id=0x5143, device_name="Adreno 830")
        q_adreno = resolve_stt_quirks(report_adreno)
        self.assertFalse(q_adreno.disable_fp16)
        self.assertFalse(q_adreno.force_medium_matmul)

    def test_installer_diffusion_destination_specs_no_overwrite(self):
        """P0-1 Fix: Verify diffusion engine bundle structure isolates libraries and prevents binary overwrites."""
        from ameva_runtime.installer import NATIVE_ASSETS, LOCAL_BIN

        self.assertIn("diffusion", NATIVE_ASSETS)
        spec = NATIVE_ASSETS["diffusion"]

        # Engine bundle isolation: primary binary is mapped to canonical name, no individual lib destination collisions
        self.assertEqual(spec.canonical_name, "sd-cli")
        self.assertEqual(spec.binary_relpath, "sd-cli-vulkan")
        self.assertEqual(LOCAL_BIN / spec.canonical_name, LOCAL_BIN / "sd-cli")

        # Verify orphan assets were purged from registry to prevent destination collisions
        self.assertNotIn("libomp", NATIVE_ASSETS)
        self.assertNotIn("libegl_shim", NATIVE_ASSETS)
        self.assertNotIn("matmul_spv", NATIVE_ASSETS)

    def test_stt_adapter_fail_fast_when_vulkan_hardware_unavailable_and_no_explicit_cpu(self):
        """Strict Zero-Silent-Fallback: If Vulkan hardware is unavailable and user did NOT request cpu, raise PlatformNotSupportedError."""
        from ameva_runtime.exceptions import PlatformNotSupportedError

        class MockEngine:
            def __init__(self):
                self.device = None
                self.threads = None

        engine = MockEngine()
        with self.assertRaises(PlatformNotSupportedError):
            SttAdapter.bind(
                engine=engine,
                profile=self.cpu_only_profile,  # non-vulkan profile
            )

    def test_installer_existing_bundle_tamper_raises_without_silent_redownload(self):
        """P0-2: Installer must raise explicit RuntimeError on tampered existing bundle, never silently re-download."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from ameva_runtime.installer import NativeAssetManager, NATIVE_ASSETS

        with tempfile.TemporaryDirectory() as td:
            fake_home = Path(td)
            fake_bin = fake_home / ".local" / "bin"
            fake_current = fake_home / ".local" / "share" / "ameva" / "current" / "stt"
            fake_bin.mkdir(parents=True)
            fake_current.mkdir(parents=True)

            canonical = fake_bin / "whisper-cli"
            canonical.write_bytes(b"tampered content")

            mgr = NativeAssetManager()
            with patch("ameva_runtime.installer.HOME", fake_home), \
                 patch("ameva_runtime.installer.LOCAL_BIN", fake_bin), \
                 patch("ameva_runtime.installer.AMEVA_CURRENT", fake_home / ".local" / "share" / "ameva" / "current"):
                with self.assertRaises(RuntimeError) as ctx:
                    mgr.provision_asset(NATIVE_ASSETS["stt"])
                self.assertIn("Broken installation", str(ctx.exception))

    def test_verify_stt_manifest_missing_raises_immediately(self):
        """P0-1/P0-2: verify_stt_manifest must never return {} silently; missing manifest must raise AmevaRuntimeError."""
        from pathlib import Path
        from ameva_runtime.adapters.stt import verify_stt_manifest
        from ameva_runtime.exceptions import AmevaRuntimeError

        non_existent_manifest = Path("/non/existent/path/manifest.json")
        with self.assertRaises(AmevaRuntimeError) as ctx:
            verify_stt_manifest(Path("/dummy/bin"), manifest_path=non_existent_manifest)
        self.assertIn("STT deployment manifest is missing", str(ctx.exception))

    def test_verify_stt_manifest_verifies_all_files_including_versioned_so(self):
        """P0-3/P0-4: Manifest verification must validate all deployed files including versioned .so.0 and not skip them."""
        import tempfile
        import json
        import hashlib
        from pathlib import Path
        from ameva_runtime.adapters.stt import verify_stt_manifest
        from ameva_runtime.exceptions import AmevaRuntimeError

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_file = root / "whisper-cli"
            bin_file.write_bytes(b"dummy binary")
            bin_sha = hashlib.sha256(b"dummy binary").hexdigest()

            # Versioned shared library (suffix == ".0", NOT ".so")
            so_versioned = root / "libggml-vulkan.so.0"
            so_versioned.write_bytes(b"tampered content")
            expected_so_sha = hashlib.sha256(b"original content").hexdigest()

            manifest_path = root / "manifest.json"
            manifest_data = {
                "bundle_id": "stt",
                "backend_feature": "vulkan",
                "target_architecture": "aarch64",
                "binary_sha256": bin_sha,
                "deployed_files": {
                    "lib/libggml-vulkan.so.0": {
                        "path": str(so_versioned),
                        "sha256": expected_so_sha,
                        "size": len(b"original content"),
                    }
                }
            }
            manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

            # Must NOT skip versioned .so.0; must raise integrity violation
            with self.assertRaises(AmevaRuntimeError) as ctx:
                verify_stt_manifest(bin_file, manifest_path=manifest_path)
            self.assertIn("Deployed file integrity violation", str(ctx.exception))

    @unittest.skipUnless(os.name == "nt", "Windows host mock test only")
    def test_windows_mock_does_not_claim_gpu_inference_verified(self):
        """P0-8: Windows mock must return HOST_MOCK_ONLY with is_gpu_verified=False, never false success."""
        import tempfile
        from pathlib import Path
        from ameva_runtime.adapters.stt import run_vulkan_inference_probe

        with tempfile.TemporaryDirectory() as td:
            vk_bin = Path(td) / "whisper-cli"
            vk_bin.write_bytes(self._make_arm64_elf(b"libvulkan.so"))
            model_file = Path(td) / "model.bin"
            model_file.write_bytes(b"model")

            res = run_vulkan_inference_probe(binary_path=vk_bin, model_path=model_file)
            self.assertEqual(res.status, "HOST_MOCK_ONLY")
            self.assertFalse(res.is_gpu_verified)
            self.assertEqual(res.backend_name, "unverified")
            self.assertFalse(res.cpu_fallback_detected)

    def test_inference_probe_execution_failure_not_classified_as_cpu_fallback(self):
        """P0-9: Execution failure or missing files must NOT be reported as cpu_fallback_detected=True."""
        from pathlib import Path
        from ameva_runtime.adapters.stt import run_vulkan_inference_probe

        res = run_vulkan_inference_probe(
            binary_path=Path("/non/existent/bin"),
            model_path=Path("/non/existent/model")
        )
        self.assertEqual(res.status, "INFERENCE_FAILED")
        self.assertFalse(res.is_gpu_verified)
        self.assertFalse(res.cpu_fallback_detected)


if __name__ == "__main__":
    unittest.main()



