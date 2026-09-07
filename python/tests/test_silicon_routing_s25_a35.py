"""
Unit & Integration Tests: Silicon-Aware Dynamic Branching (Galaxy S25 vs Galaxy A35)
===================================================================================
Verifies that v2.0.0 correctly discriminates between Qualcomm Adreno 830 (Galaxy S25)
and ARM Mali-G68 MP5 (Galaxy A35), applying appropriate pipeline routing and quirks
with zero cross-contamination.
"""
import unittest
from ameva_runtime.vulkan.doctor import Doctor
from ameva_runtime.detector import HardwareProfile
from ameva_runtime.router import SmartRouter


class TestSiliconRoutingS25AndA35(unittest.TestCase):
    def setUp(self):
        self.doc = Doctor()

    def test_galaxy_s25_profile_matching(self):
        """Verify Galaxy S25 (Snapdragon 8 Elite / Adreno 830) profile detection."""
        # 1. Exact model match
        prof_model = self.doc.load_hardware_profile("SM-S931N", 0x5143)
        self.assertEqual(prof_model.get("model"), "SM-S931N")
        self.assertEqual(prof_model.get("market_name"), "Samsung Galaxy S25")
        self.assertTrue(prof_model.get("subgroup_control_bypass"))
        self.assertFalse(prof_model.get("enforce_medium_matmul", False))

        # 2. GPU name match
        prof_gpu = self.doc.load_hardware_profile("Adreno 830", 0x5143)
        self.assertEqual(prof_gpu.get("model"), "SM-S931N")
        self.assertTrue(prof_gpu.get("subgroup_control_bypass"))
        self.assertFalse(prof_gpu.get("enforce_medium_matmul", False))

    def test_galaxy_a35_profile_matching(self):
        """Verify Galaxy A35 (Exynos 1380 / Mali-G68 MP5) profile detection."""
        # 1. Exact model match
        prof_model = self.doc.load_hardware_profile("SM-A356N", 0x13B5)
        self.assertEqual(prof_model.get("model"), "SM-A356N")
        self.assertEqual(prof_model.get("market_name"), "Samsung Galaxy A35 5G")
        self.assertTrue(prof_model.get("enforce_medium_matmul"))
        self.assertEqual(prof_model.get("memory_alignment_bytes"), 128)

        # 2. GPU name match
        prof_gpu = self.doc.load_hardware_profile("Mali-G68", 0x13B5)
        self.assertEqual(prof_gpu.get("model"), "SM-A356N")
        self.assertTrue(prof_gpu.get("enforce_medium_matmul"))

    def test_s25_router_execution_plan(self):
        """Verify SmartRouter produces high-performance Vulkan plan for Galaxy S25 without Mali quirks."""
        s25_profile = HardwareProfile(
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
            diagnosis_reason="Snapdragon 8 Elite Adreno 830 Vulkan Active",
        )
        router = SmartRouter(s25_profile)
        plan = router.route_for_llm("qwen2.5-7b")

        self.assertEqual(plan.backend, "vulkan")
        self.assertTrue(plan.is_gpu_accelerated)
        self.assertGreaterEqual(plan.ngl, 30)
        self.assertNotIn("GGML_VK_FORCE_MEDIUM_MATMUL", plan.env_overrides)
        self.assertIn("ADRENO", plan.diagnosis)

    def test_a35_router_execution_plan(self):
        """Verify SmartRouter produces zero-freeze Medium MatMul Vulkan plan for Galaxy A35."""
        a35_profile = HardwareProfile(
            vendor="samsung_exynos",
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
            diagnosis_reason="Exynos 1380 Mali-G68 Vulkan Active",
        )
        router = SmartRouter(a35_profile)
        plan = router.route_for_llm("qwen2.5-0.5b")

        self.assertEqual(plan.backend, "vulkan")
        self.assertTrue(plan.is_gpu_accelerated)
        self.assertGreaterEqual(plan.ngl, 20)
        # Critical verification: GGML_VK_FORCE_MEDIUM_MATMUL must be present for Mali
        self.assertEqual(plan.env_overrides.get("GGML_VK_FORCE_MEDIUM_MATMUL"), "1")
        self.assertEqual(plan.env_overrides.get("GGML_VK_DISABLE_F16"), "1")
        self.assertIn("MALI", plan.diagnosis)
        self.assertIn("Medium MatMul", plan.diagnosis)

    def test_no_cross_contamination(self):
        """Verify that Mali quirks never leak into Adreno, and Adreno quirks never leak into Mali."""
        # Generic Adreno
        adreno_prof = self.doc.load_hardware_profile("Qualcomm Generic Adreno", 0x5143)
        self.assertFalse(adreno_prof.get("enforce_medium_matmul", False))

        # Generic Mali
        mali_prof = self.doc.load_hardware_profile("ARM Generic Mali", 0x13B5)
        self.assertTrue(mali_prof.get("enforce_medium_matmul", False))
        self.assertFalse(mali_prof.get("subgroup_control_bypass", False))

    def test_a35_tts_routing(self):
        """Verify SmartRouter produces proper ExecutionPlan for TTS on Galaxy A35."""
        from ameva_runtime.adapters.tts import TtsAdapter
        a35_profile = HardwareProfile(
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
            diagnosis_reason="Exynos 1380 Mali-G68 Active",
        )
        router = SmartRouter(a35_profile)
        # 1. Vulkan route
        vk_plan = router.route_for_tts(requested_backend="vulkan")
        self.assertEqual(vk_plan.backend, "vulkan")
        self.assertTrue(vk_plan.is_gpu_accelerated)
        self.assertEqual(vk_plan.threads, 4)
        self.assertEqual(vk_plan.env_overrides.get("AMEVA_VK_DSP_ACCEL"), "1")
        self.assertIn("--device", vk_plan.cli_flags)
        self.assertIn("gpu", vk_plan.cli_flags)

        # 2. CPU route
        cpu_plan = router.route_for_tts(requested_backend="cpu")
        self.assertEqual(cpu_plan.backend, "cpu_neon")
        self.assertFalse(cpu_plan.is_gpu_accelerated)
        self.assertEqual(cpu_plan.threads, 4)

        # 3. TtsAdapter binding
        class MockEngine:
            device = "auto"
            threads = 1
            backend = "unknown"

        mock_eng = MockEngine()
        binding = TtsAdapter.bind(engine=mock_eng, profile=a35_profile, requested_backend="vulkan")
        self.assertEqual(mock_eng.device, "vulkan")
        self.assertEqual(mock_eng.threads, 4)
        self.assertEqual(mock_eng.backend, "vulkan")
        self.assertEqual(binding.status, "BOUND_VULKAN")


if __name__ == "__main__":
    unittest.main()
