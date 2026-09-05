"""
Unit Tests: STT (Whisper) Hardware-Aware Adaptive Routing & Quirks
===================================================================
Verifies that SmartRouter and SttAdapter produce optimal Vulkan plans
for Galaxy S25 (Adreno 830) and Galaxy A35 (Mali-G68) with correct
Medium MatMul quirks, library paths, and clean CPU fallbacks.
"""
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

    def test_stt_adapter_bind_vulkan(self):
        """SttAdapter.bind should produce BOUND_VULKAN status and config on Vulkan device."""
        class MockEngine:
            def __init__(self):
                self.device = None
                self.threads = None

        engine = MockEngine()
        res = SttAdapter.bind(engine=engine, profile=self.a35_profile)

        self.assertEqual(res.status, "BOUND_VULKAN")
        self.assertEqual(res.backend, "vulkan")
        self.assertEqual(engine.device, "vulkan")
        self.assertEqual(engine.threads, 4)
        self.assertIn("env_overrides", res.config)
        self.assertEqual(res.config["env_overrides"].get("GGML_VK_FORCE_MEDIUM_MATMUL"), "1")

    def test_stt_adapter_bind_cpu_fallback(self):
        """SttAdapter.bind should produce BOUND_CPU_NEON status on non-Vulkan device."""
        class MockEngine:
            def __init__(self):
                self.device = None
                self.threads = None

        engine = MockEngine()
        res = SttAdapter.bind(engine=engine, profile=self.cpu_only_profile)

        self.assertEqual(res.status, "BOUND_CPU_NEON")
        self.assertEqual(res.backend, "cpu_neon")
        self.assertEqual(engine.device, "cpu")
        self.assertEqual(engine.threads, 2)


if __name__ == "__main__":
    unittest.main()
