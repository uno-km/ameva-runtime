"""Unit tests for Python Native Asset Installer Security (Path Traversal and SHA-256 Checksum Enforcement)."""

import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import unittest
from pathlib import Path

from ameva_runtime.installer import (
    safe_extract_tar,
    verify_sha256,
    AssetSpec,
    NativeAssetManager,
)


class TestInstallerSecurity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_tar_gz(self, members_data: dict, symlinks: dict = None, hardlinks: dict = None) -> Path:
        tar_path = self.root / "test_archive.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for name, content in members_data.items():
                data_bytes = content.encode("utf-8") if isinstance(content, str) else content
                ti = tarfile.TarInfo(name=name)
                ti.size = len(data_bytes)
                ti.type = tarfile.REGTYPE
                tar.addfile(ti, io.BytesIO(data_bytes))
            if symlinks:
                for link_name, target in symlinks.items():
                    ti = tarfile.TarInfo(name=link_name)
                    ti.type = tarfile.SYMTYPE
                    ti.linkname = target
                    tar.addfile(ti)
            if hardlinks:
                for link_name, target in hardlinks.items():
                    ti = tarfile.TarInfo(name=link_name)
                    ti.type = tarfile.LNKTYPE
                    ti.linkname = target
                    tar.addfile(ti)
        return tar_path

    # P0: Archive Traversal Tests
    def test_safe_extract_tar_rejects_parent_traversal(self):
        archive = self._create_tar_gz({"../../escaped.txt": "EVIL"})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("escapes target directory", str(ctx.exception))

    def test_safe_extract_tar_rejects_deep_parent_traversal(self):
        archive = self._create_tar_gz({"foo/bar/../../../../escaped.txt": "EVIL"})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("escapes target directory", str(ctx.exception))

    def test_safe_extract_tar_rejects_symlink(self):
        archive = self._create_tar_gz(
            {"target.txt": "SAFE"},
            symlinks={"evil_link": "../../outside"}
        )
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("Archive links are not permitted", str(ctx.exception))

    def test_safe_extract_tar_rejects_hardlink(self):
        archive = self._create_tar_gz(
            {"target.txt": "SAFE"},
            hardlinks={"evil_hardlink": "target.txt"}
        )
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("Archive links are not permitted", str(ctx.exception))

    def test_safe_extract_tar_extracts_legitimate_files(self):
        archive = self._create_tar_gz({
            "bin/tool": "TOOL_BINARY_CONTENT",
            "lib/libfoo.so": "SHARED_OBJECT_CONTENT",
        })
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        safe_extract_tar(archive, target_dir)
        self.assertTrue((target_dir / "bin" / "tool").exists())
        self.assertEqual((target_dir / "bin" / "tool").read_text(), "TOOL_BINARY_CONTENT")
        self.assertTrue((target_dir / "lib" / "libfoo.so").exists())
        self.assertEqual((target_dir / "lib" / "libfoo.so").read_text(), "SHARED_OBJECT_CONTENT")

    # P1: SHA-256 Checksum Enforcement Tests
    def test_verify_sha256_valid(self):
        test_file = self.root / "sample.bin"
        test_file.write_bytes(b"HELLO_WORLD_TEST")
        expected_hash = hashlib.sha256(b"HELLO_WORLD_TEST").hexdigest()
        verify_sha256(test_file, expected_hash)
        self.assertTrue(test_file.exists())

    def test_verify_sha256_mismatch_unlinks_file(self):
        test_file = self.root / "sample_tampered.bin"
        test_file.write_bytes(b"TAMPERED_DATA")
        wrong_hash = "0" * 64
        with self.assertRaises(RuntimeError) as ctx:
            verify_sha256(test_file, wrong_hash)
        self.assertIn("SHA-256 mismatch", str(ctx.exception))
        self.assertFalse(test_file.exists())

    def test_verify_sha256_missing_hash_rejected(self):
        test_file = self.root / "sample_no_hash.bin"
        test_file.write_bytes(b"DATA")
        with self.assertRaises(RuntimeError) as ctx:
            verify_sha256(test_file, None)
        self.assertIn("Missing expected SHA-256", str(ctx.exception))
        self.assertFalse(test_file.exists())

    def test_verify_sha256_invalid_hex_rejected(self):
        test_file = self.root / "sample_bad_hex.bin"
        test_file.write_bytes(b"DATA")
        with self.assertRaises(RuntimeError) as ctx:
            verify_sha256(test_file, "INVALID_HEX_NOT_64_CHARS")
        self.assertIn("Invalid expected SHA-256", str(ctx.exception))
        self.assertFalse(test_file.exists())

    def test_provision_asset_missing_sha256_fails_fast(self):
        spec = AssetSpec(
            name="test_missing_sha",
            filename="nonexistent.bin",
            target_path=self.root / "bin" / "test_bin",
            sha256=None,
        )
        mgr = NativeAssetManager()
        success = mgr.provision_asset(spec)
        self.assertFalse(success)
        self.assertFalse(spec.target_path.exists())


if __name__ == "__main__":
    unittest.main()
