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
            # Buffer allocation within limits
            buf = ctx.allocate_buffer(1024 * 1024 * 100)  # 100MB
            self.assertEqual(buf, 1024 * 1024 * 100)

        self.assertFalse(ctx._is_active)

    def test_buffer_overflow_rejection(self):
        with create_context(device="auto", memory_limit_mb=64) as ctx:
            with self.assertRaises(BufferAllocationError):
                ctx.allocate_buffer(1024 * 1024 * 128)  # 128MB > 64MB limit

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


if __name__ == "__main__":
    unittest.main()
