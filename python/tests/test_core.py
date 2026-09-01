"""
VulkanContext core 테스트 — bind() API 기반.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ameva_vulkan_runtime.core import create_context, VulkanContext
from ameva_vulkan_runtime.exceptions import BufferAllocationError


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


if __name__ == "__main__":
    unittest.main()
