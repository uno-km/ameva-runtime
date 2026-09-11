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

    def _create_tar_gz(self, members_data: dict, symlinks: dict = None, hardlinks: dict = None, special_members: dict = None) -> Path:
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
            if special_members:
                for spec_name, member_type in special_members.items():
                    ti = tarfile.TarInfo(name=spec_name)
                    ti.type = member_type
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

    # Resource Exhaustion & Special Member Rejection Tests
    def test_safe_extract_tar_rejects_fifo(self):
        archive = self._create_tar_gz({"safe.txt": "SAFE"}, special_members={"evil_fifo": tarfile.FIFOTYPE})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("FIFO_REJECTED", str(ctx.exception))

    def test_safe_extract_tar_rejects_char_device(self):
        archive = self._create_tar_gz({"safe.txt": "SAFE"}, special_members={"evil_chr": tarfile.CHRTYPE})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("CHAR_DEVICE_REJECTED", str(ctx.exception))

    def test_safe_extract_tar_rejects_block_device(self):
        archive = self._create_tar_gz({"safe.txt": "SAFE"}, special_members={"evil_blk": tarfile.BLKTYPE})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("BLOCK_DEVICE_REJECTED", str(ctx.exception))

    def test_safe_extract_tar_rejects_special_member(self):
        archive = self._create_tar_gz({"safe.txt": "SAFE"}, special_members={"evil_special": b"?"})
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir)
        self.assertIn("SPECIAL_MEMBER_REJECTED", str(ctx.exception))

    def test_safe_extract_tar_rejects_duplicate_path(self):
        tar_path = self.root / "dup_archive.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for _ in range(2):
                ti = tarfile.TarInfo(name="duplicate.txt")
                ti.size = 4
                ti.type = tarfile.REGTYPE
                tar.addfile(ti, io.BytesIO(b"DATA"))
        target_dir = self.root / "extracted"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(tar_path, target_dir)
        self.assertIn("DUPLICATE_PATH_REJECTED", str(ctx.exception))

    # Mandatory Exact Boundary Enforcement Tests (8 Gates)
    def test_file_count_exact_limit_accepted(self):
        archive = self._create_tar_gz({"f1.txt": "1", "f2.txt": "2", "f3.txt": "3"})
        target_dir = self.root / "extracted_exact_fc"
        target_dir.mkdir()
        safe_extract_tar(archive, target_dir, max_file_count=3)
        self.assertTrue((target_dir / "f1.txt").exists())
        self.assertTrue((target_dir / "f2.txt").exists())
        self.assertTrue((target_dir / "f3.txt").exists())

    def test_file_count_limit_plus_one_rejected(self):
        archive = self._create_tar_gz({"f1.txt": "1", "f2.txt": "2", "f3.txt": "3", "f4.txt": "4"})
        target_dir = self.root / "extracted_plus1_fc"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir, max_file_count=3)
        self.assertIn("MAX_FILE_COUNT_ENFORCED", str(ctx.exception))

    def test_single_file_exact_limit_accepted(self):
        archive = self._create_tar_gz({"exact.bin": "A" * 100})
        target_dir = self.root / "extracted_exact_sfs"
        target_dir.mkdir()
        safe_extract_tar(archive, target_dir, max_single_file_size=100)
        self.assertEqual(len((target_dir / "exact.bin").read_text()), 100)

    def test_single_file_limit_plus_one_rejected(self):
        archive = self._create_tar_gz({"plus1.bin": "A" * 101})
        target_dir = self.root / "extracted_plus1_sfs"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir, max_single_file_size=100)
        self.assertIn("MAX_SINGLE_FILE_SIZE_ENFORCED", str(ctx.exception))

    def test_total_size_exact_limit_accepted(self):
        archive = self._create_tar_gz({"a.bin": "A" * 50, "b.bin": "B" * 50})
        target_dir = self.root / "extracted_exact_ts"
        target_dir.mkdir()
        safe_extract_tar(archive, target_dir, max_total_size=100)
        self.assertEqual(len((target_dir / "a.bin").read_text()), 50)
        self.assertEqual(len((target_dir / "b.bin").read_text()), 50)

    def test_total_size_limit_plus_one_rejected(self):
        archive = self._create_tar_gz({"a.bin": "A" * 50, "b.bin": "B" * 51})
        target_dir = self.root / "extracted_plus1_ts"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir, max_total_size=100)
        self.assertIn("MAX_TOTAL_EXPANDED_SIZE_ENFORCED", str(ctx.exception))

    def test_path_depth_exact_limit_accepted(self):
        archive = self._create_tar_gz({"a/b/c.txt": "EXACT_DEPTH"})
        target_dir = self.root / "extracted_exact_depth"
        target_dir.mkdir()
        safe_extract_tar(archive, target_dir, max_path_depth=3)
        self.assertTrue((target_dir / "a" / "b" / "c.txt").exists())

    def test_path_depth_limit_plus_one_rejected(self):
        archive = self._create_tar_gz({"a/b/c/d.txt": "PLUS1_DEPTH"})
        target_dir = self.root / "extracted_plus1_depth"
        target_dir.mkdir()
        with self.assertRaises(RuntimeError) as ctx:
            safe_extract_tar(archive, target_dir, max_path_depth=3)
        self.assertIn("MAX_PATH_DEPTH_ENFORCED", str(ctx.exception))

    def test_production_limit_constants_match_policy(self):
        import inspect
        from ameva_runtime.installer import (
            MAX_ARCHIVE_FILE_COUNT,
            MAX_ARCHIVE_SINGLE_FILE_SIZE,
            MAX_ARCHIVE_TOTAL_EXPANDED_SIZE,
            MAX_ARCHIVE_PATH_DEPTH,
            safe_extract_tar,
        )
        self.assertEqual(MAX_ARCHIVE_FILE_COUNT, 1000)
        self.assertEqual(MAX_ARCHIVE_SINGLE_FILE_SIZE, 250 * 1024 * 1024)
        self.assertEqual(MAX_ARCHIVE_TOTAL_EXPANDED_SIZE, 500 * 1024 * 1024)
        self.assertEqual(MAX_ARCHIVE_PATH_DEPTH, 16)

        sig = inspect.signature(safe_extract_tar)
        self.assertEqual(sig.parameters["max_file_count"].default, MAX_ARCHIVE_FILE_COUNT)
        self.assertEqual(sig.parameters["max_single_file_size"].default, MAX_ARCHIVE_SINGLE_FILE_SIZE)
        self.assertEqual(sig.parameters["max_total_size"].default, MAX_ARCHIVE_TOTAL_EXPANDED_SIZE)
        self.assertEqual(sig.parameters["max_path_depth"].default, MAX_ARCHIVE_PATH_DEPTH)


if __name__ == "__main__":
    unittest.main()
