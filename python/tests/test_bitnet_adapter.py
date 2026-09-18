"""Unit tests for BitnetAdapter and $PREFIX/bin SSOT resolution."""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ameva_runtime.adapters.bitnet import BitnetAdapter
from ameva_runtime.exceptions import AmevaRuntimeError, AmevaLlamaAssetMissingError, AmevaLlamaVerificationError


class TestBitnetAdapter(unittest.TestCase):
    def test_resolve_binary_path_not_found_raises_missing_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(Path(tmpdir) / "usr")}):
                with self.assertRaises(AmevaLlamaAssetMissingError):
                    BitnetAdapter.resolve_binary_path()

    def test_resolve_binary_path_manifest_missing_raises_verification_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "termux-bitnet-cli"
            bin_file.write_bytes(b"ELF_FAKE_BITNET_BIN")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                with self.assertRaises(AmevaLlamaVerificationError):
                    BitnetAdapter.resolve_binary_path()

    def test_resolve_binary_path_hash_mismatch_raises_verification_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "termux-bitnet-cli"
            bin_file.write_bytes(b"ELF_FAKE_BITNET_BIN")

            manifest_dir = Path(tmpdir) / ".local" / "share" / "ameva" / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / "bitnet.json"
            manifest_file.write_text(json.dumps({
                "bundle_id": "bitnet",
                "binary_sha256": "0" * 64,
            }), encoding="utf-8")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                with self.assertRaises(AmevaLlamaVerificationError):
                    BitnetAdapter.resolve_binary_path()

    def test_resolve_binary_path_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            bin_dir = prefix_dir / "bin"
            bin_dir.mkdir(parents=True, exist_ok=True)
            bin_file = bin_dir / "termux-bitnet-cli"
            content = b"ELF_VALID_BITNET_CLI"
            bin_file.write_bytes(content)
            bin_sha = hashlib.sha256(content).hexdigest()

            manifest_dir = Path(tmpdir) / ".local" / "share" / "ameva" / "manifests"
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_file = manifest_dir / "bitnet.json"
            manifest_file.write_text(json.dumps({
                "bundle_id": "bitnet",
                "binary_sha256": bin_sha,
            }), encoding="utf-8")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                resolved = BitnetAdapter.resolve_binary_path()
                self.assertEqual(resolved, str(bin_file.resolve()))

    def test_resolve_library_path_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            lib_dir = prefix_dir / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            so_file = lib_dir / "libtermux_bitnet.so"
            so_file.write_bytes(b"ELF_VALID_SO")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                resolved = BitnetAdapter.resolve_library_path()
                self.assertEqual(resolved, str(so_file.resolve()))

    def test_resolve_library_path_fallback_to_isolated_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            prefix_dir.mkdir(parents=True, exist_ok=True)
            isolated_lib = Path(tmpdir) / ".local" / "share" / "ameva" / "current" / "bitnet" / "lib"
            isolated_lib.mkdir(parents=True, exist_ok=True)
            so_file = isolated_lib / "libtermux_bitnet.so"
            so_file.write_bytes(b"ELF_ISOLATED_SO")

            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                resolved = BitnetAdapter.resolve_library_path()
                self.assertEqual(resolved, str(so_file.resolve()))

    def test_resolve_library_path_missing_raises_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix_dir = Path(tmpdir) / "usr"
            with patch.dict("os.environ", {"HOME": tmpdir, "PREFIX": str(prefix_dir)}):
                with self.assertRaises(AmevaRuntimeError):
                    BitnetAdapter.resolve_library_path()


if __name__ == "__main__":
    unittest.main()
