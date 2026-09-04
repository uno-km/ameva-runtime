"""
VulkanContext core 테스트 — bind() API 기반.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ameva_vulkan_runtime.core import create_context, get_or_create_context, VulkanContext
from ameva_vulkan_runtime.exceptions import BufferAllocationError, PlatformNotSupportedError


class TestCoreContext(unittest.TestCase):
    def test_auto_context_lifecycle(self):
        with create_context(device="auto", memory_limit_mb=512) as ctx:
            self.assertTrue(ctx._is_active)
            # Buffer allocation within limits (real bytearray allocated)
            buf = ctx.allocate_buffer(1024 * 1024 * 10)  # 10MB
            self.assertIsInstance(buf, bytearray)
            self.assertEqual(len(buf), 1024 * 1024 * 10)
            self.assertTrue(ctx.validate_buffer_budget(1024 * 1024 * 100))

        self.assertFalse(ctx._is_active)

    def test_buffer_overflow_rejection(self):
        with create_context(device="auto", memory_limit_mb=64) as ctx:
            with self.assertRaises(BufferAllocationError):
                ctx.validate_buffer_budget(1024 * 1024 * 128)  # 128MB > 64MB limit

    def test_context_exposes_backend_type(self):
        """backend_type 이 'vulkan' 또는 'cpu_neon' 중 하나인지 검증."""
        with create_context(device="auto") as ctx:
            self.assertIn(ctx.backend_type, ("vulkan", "cpu_neon"))

    def test_is_vulkan_consistent_with_backend(self):
        """is_vulkan() 메서드가 backend_type 과 일관됨을 검증."""
        with create_context(device="auto") as ctx:
            self.assertEqual(ctx.is_vulkan(), ctx.backend_type == "vulkan")
            self.assertEqual(ctx.is_gpu, ctx.backend_type == "vulkan")

    def test_explicit_cpu_mode_bypass(self):
        """device='cpu' 지정 시 Vulkan 바이패스 및 CPU NEON 모드로 즉시 초기화됨을 검증."""
        with create_context(device="cpu") as ctx:
            self.assertFalse(ctx.is_gpu)
            self.assertEqual(ctx.backend_type, "cpu_neon")
            self.assertIn("NEON", ctx.device_name)
            flags = ctx.to_engine_flags("whisper")
            self.assertFalse(flags["use_gpu"])

    def test_get_or_create_context_reuse(self):
        """기생성된 VulkanContext 전달 시 재사용 확인."""
        ctx1 = create_context(device="cpu")
        ctx2 = get_or_create_context(ctx1)
        self.assertIs(ctx1, ctx2)
        ctx1.close()

    def test_get_or_create_context_from_string(self):
        """문자열 전달 시 올바른 컨텍스트 반환 검증."""
        ctx = get_or_create_context("cpu")
        self.assertEqual(ctx.backend_type, "cpu_neon")
        ctx.close()

    def test_soc_mali_termux_auto_routing(self):
        """Mali GPU in Termux CLI must automatically and immediately route to CPU NEON without Vulkan probe."""
        from unittest.mock import patch
        from ameva_vulkan_runtime.platform import SoCInfo

        mock_soc = SoCInfo(
            vendor="samsung_exynos",
            chipname="exynos1380",
            gpu_family="mali",
            kgsl_accessible=False,
            mali_node_accessible=True,
            can_direct_vulkan_cli=False,
            recommended_backend="cpu_neon",
            cpu_model="Cortex-A78 x4 + Cortex-A55 x4",
            cpu_cores=8,
            diagnosis_reason="Samsung Exynos/Mali in Termux CLI lacks headless window context. Routed to pure ARM NEON CPU mode.",
        )

        with patch("ameva_vulkan_runtime.core.detect_soc_environment", return_value=mock_soc):
            with create_context(device="auto") as ctx:
                self.assertEqual(ctx.backend_type, "cpu_neon")
                self.assertFalse(ctx.is_gpu)
                self.assertIn("NEON", ctx.device_name)
                self.assertEqual(ctx.execution_flags["soc_vendor"], "samsung_exynos")

    def test_soc_mali_explicit_gpu_fail_fast(self):
        """Mali GPU in Termux CLI requesting explicit 'vulkan' or 'gpu' must raise PlatformNotSupportedError (Zero-Silent-Fallback)."""
        from unittest.mock import patch
        from ameva_vulkan_runtime.platform import SoCInfo

        mock_soc = SoCInfo(
            vendor="samsung_exynos",
            chipname="exynos1380",
            gpu_family="mali",
            kgsl_accessible=False,
            mali_node_accessible=True,
            can_direct_vulkan_cli=False,
            recommended_backend="cpu_neon",
            cpu_model="Cortex-A78 x4 + Cortex-A55 x4",
            cpu_cores=8,
            diagnosis_reason="Samsung Exynos/Mali in Termux CLI lacks headless window context.",
        )

        with patch("ameva_vulkan_runtime.core.detect_soc_environment", return_value=mock_soc):
            with self.assertRaises(PlatformNotSupportedError) as cm:
                create_context(device="vulkan")
            self.assertIn("Zero-Silent-Fallback", str(cm.exception))
            self.assertIn("Mali", str(cm.exception))

    def test_soc_adreno_vulkan_routing(self):
        """Qualcomm Adreno with accessible KGSL allows direct Vulkan probe."""
        from unittest.mock import patch
        from ameva_vulkan_runtime.platform import SoCInfo
        from ameva_vulkan_runtime.doctor import Doctor

        mock_soc = SoCInfo(
            vendor="qualcomm",
            chipname="sm8650",
            gpu_family="adreno",
            kgsl_accessible=True,
            mali_node_accessible=False,
            can_direct_vulkan_cli=True,
            recommended_backend="vulkan",
            cpu_model="Cortex-X4 + Cortex-A720",
            cpu_cores=8,
            diagnosis_reason="Qualcomm Adreno with accessible /dev/kgsl-3d0 node.",
        )

        with patch("ameva_vulkan_runtime.core.detect_soc_environment", return_value=mock_soc):
            with patch.object(Doctor, "quick_probe", return_value=True):
                with patch.object(Doctor, "quick_probe_device", return_value="Adreno (TM) 750"):
                    with create_context(device="auto") as ctx:
                        self.assertEqual(ctx.backend_type, "vulkan")
                        self.assertTrue(ctx.is_gpu)
                        self.assertEqual(ctx.device_name, "Adreno (TM) 750")


if __name__ == "__main__":
    unittest.main()

