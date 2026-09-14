"""
Gate 1 Quality Verification Test Suite: Loader Safety & Namespace Integrity
=============================================================================
Enforces OpenSSF/CNCF compliance:
1. test_env_does_not_inject_system_lib64:
   Verifies get_vulkan_env() strictly excludes /system/lib64, /vendor/lib64,
   /apex/..., and other Android system paths from LD_LIBRARY_PATH.
2. test_env_preserves_unrelated_user_variables:
   Verifies get_vulkan_env() does not strip or alter unrelated user environment
   variables or valid user paths.
3. test_no_vulkan_symlink_bridge_required:
   Verifies get_vulkan_env() executes without creating, requiring, or modifying
   symlinks in ~/.local/share/ameva/lib.
4. test_loader_failure_propagates:
   Verifies VulkanDynamicLoader fails closed (raises AmevaVulkanError) when
   encountering non-existent or corrupted shared objects.
5. test_loader_handle_outlives_dispatch_table:
   Verifies VulkanDispatchTable is strictly bound to VulkanDynamicLoader lifetime,
   and raises AmevaVulkanError if invoked after loader unload.
6. test_missing_vkGetInstanceProcAddr_fails_closed:
   Verifies that if a library lacks vkGetInstanceProcAddr, VulkanDynamicLoader
   fails closed immediately and cleans up its handle.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ameva_runtime.adapters.base import get_vulkan_env
from ameva_runtime.vulkan.loader import VulkanDynamicLoader, VulkanDispatchTable
from ameva_runtime.vulkan.exceptions import AmevaVulkanError


class TestGate1LoaderSafety(unittest.TestCase):
    def test_env_does_not_inject_system_lib64(self):
        """LD_LIBRARY_PATH must never contain /system/lib64, /vendor/lib64, or /apex."""
        # Case 1: Base env without LD_LIBRARY_PATH
        env1 = get_vulkan_env({})
        ld1 = env1.get("LD_LIBRARY_PATH", "")
        for forbidden in ("/system/", "/vendor/", "/apex/", "/system_ext/", "/odm/", "/product/"):
            self.assertNotIn(forbidden, ld1, f"Forbidden system prefix '{forbidden}' detected in LD_LIBRARY_PATH: {ld1}")

        # Case 2: Base env with user LD_LIBRARY_PATH containing both valid and forbidden paths
        dirty_env = {
            "LD_LIBRARY_PATH": "/custom/valid/lib:/system/lib64:/vendor/lib64:/another/valid/lib:/apex/com.android.runtime/lib64"
        }
        env2 = get_vulkan_env(dirty_env)
        ld2 = env2.get("LD_LIBRARY_PATH", "")
        self.assertIn("/custom/valid/lib", ld2)
        self.assertIn("/another/valid/lib", ld2)
        for forbidden in ("/system/lib64", "/vendor/lib64", "/apex/com.android.runtime/lib64"):
            self.assertNotIn(forbidden, ld2, f"Dirty system path '{forbidden}' was not filtered out: {ld2}")

    def test_env_preserves_unrelated_user_variables(self):
        """User environment variables must be preserved intact without modification."""
        user_env = {
            "CUSTOM_API_KEY": "secret_token_12345",
            "PORT": "8080",
            "USER_TAG": "production-node-alpha",
            "CUDA_VISIBLE_DEVICES": "none",
            "LD_LIBRARY_PATH": "/opt/custom/lib",
        }
        env = get_vulkan_env(user_env)
        self.assertEqual(env["CUSTOM_API_KEY"], "secret_token_12345")
        self.assertEqual(env["PORT"], "8080")
        self.assertEqual(env["USER_TAG"], "production-node-alpha")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "none")
        self.assertIn("/opt/custom/lib", env["LD_LIBRARY_PATH"])

    def test_no_vulkan_symlink_bridge_required(self):
        """No Central Bridge symlink (~/.local/share/ameva/lib/libvulkan.so) is created or queried."""
        with patch("pathlib.Path.home") as mock_home:
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                mock_home.return_value = Path(temp_dir)
                env = get_vulkan_env({})
                bridge_dir = Path(temp_dir) / ".local" / "share" / "ameva" / "lib"
                if bridge_dir.exists():
                    symlinks = [p for p in bridge_dir.iterdir() if p.is_symlink()]
                    self.assertEqual(len(symlinks), 0, f"Unexpected symlinks created in bridge: {symlinks}")
                self.assertNotIn(str(bridge_dir), env.get("LD_LIBRARY_PATH", ""))

    def test_loader_failure_propagates(self):
        """Dynamic loader must fail closed (raise AmevaVulkanError) when library loading fails."""
        loader = VulkanDynamicLoader(explicit_path="/nonexistent/path/to/libvulkan_fake.so")
        with self.assertRaises(AmevaVulkanError) as ctx:
            loader.load()
        self.assertIn("Gate 1 Fail-Closed", str(ctx.exception))
        self.assertFalse(loader.is_loaded())

    def test_loader_handle_outlives_dispatch_table(self):
        """Dispatch table is bound to loader lifecycle and fails closed if loader is unloaded."""
        loader = VulkanDynamicLoader()
        mock_handle = MagicMock()
        mock_gpa = MagicMock(return_value=0x12345678)
        loader._handle = mock_handle
        loader._loaded_path = "mock_libvulkan.so"
        loader._vkGetInstanceProcAddr = mock_gpa

        self.assertTrue(loader.is_loaded())
        table = loader.create_dispatch_table()

        addr = table.get_proc_addr("vkCreateInstance")
        self.assertEqual(addr, 0x12345678)

        loader.unload()
        self.assertFalse(loader.is_loaded())

        with self.assertRaises(AmevaVulkanError) as ctx:
            table.get_proc_addr("vkCreateInstance")
        self.assertIn("Loader handle has been unloaded", str(ctx.exception))

        with self.assertRaises(AmevaVulkanError) as ctx:
            _ = table.vkCreateInstance
        self.assertIn("Loader handle has been unloaded", str(ctx.exception))

    def test_missing_vkGetInstanceProcAddr_fails_closed(self):
        """If library lacks vkGetInstanceProcAddr, loader fails closed and unloads."""
        loader = VulkanDynamicLoader()
        mock_lib = MagicMock(spec=[])

        with patch("ctypes.CDLL", return_value=mock_lib):
            with self.assertRaises(AmevaVulkanError) as ctx:
                loader.load(explicit_path="mock_no_gpa.so")
            self.assertIn("vkGetInstanceProcAddr symbol missing", str(ctx.exception))
            self.assertFalse(loader.is_loaded())
            self.assertIsNone(loader._handle)


if __name__ == "__main__":
    unittest.main()
