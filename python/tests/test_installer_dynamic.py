"""Unit tests for dynamic URL resolution and zero-hardcoding in ameva_runtime.installer."""

import os
import unittest
from unittest.mock import patch

from ameva_runtime._version import __version__
from ameva_runtime.installer import (
    get_release_base_url,
    NativeAssetManager,
    GITHUB_RELEASE_LATEST,
    NATIVE_ASSETS,
)


class TestInstallerDynamicResolution(unittest.TestCase):
    def test_default_release_url_matches_current_version(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AMEVA_RELEASE_BASE", None)
            os.environ.pop("AMEVA_RELEASE_TAG", None)
            url = get_release_base_url()
            self.assertEqual(
                url,
                f"https://github.com/uno-km/ameva-runtime/releases/download/v{__version__}",
            )

    def test_env_var_release_tag_override(self):
        with patch.dict(os.environ, {"AMEVA_RELEASE_TAG": "v2.9.9"}):
            url = get_release_base_url()
            self.assertEqual(
                url,
                "https://github.com/uno-km/ameva-runtime/releases/download/v2.9.9",
            )

        with patch.dict(os.environ, {"AMEVA_RELEASE_TAG": "2.9.9"}):
            url = get_release_base_url()
            self.assertEqual(
                url,
                "https://github.com/uno-km/ameva-runtime/releases/download/v2.9.9",
            )

    def test_env_var_release_base_override(self):
        custom = "https://custom-mirror.example.com/releases/assets"
        with patch.dict(os.environ, {"AMEVA_RELEASE_BASE": custom}):
            url = get_release_base_url()
            self.assertEqual(url, custom)

    def test_native_asset_manager_defaults_to_dynamic_url(self):
        mgr = NativeAssetManager()
        self.assertIn(f"v{__version__}", mgr.base_url)

    def test_native_assets_registry_completeness(self):
        self.assertIn("diffusion", NATIVE_ASSETS)
        self.assertIn("stt", NATIVE_ASSETS)
        self.assertIn("tts", NATIVE_ASSETS)
        self.assertIn("libomp", NATIVE_ASSETS)
        self.assertIn("libegl_shim", NATIVE_ASSETS)
        self.assertIn("matmul_spv", NATIVE_ASSETS)
