"""
AMEVA Native Asset Provisioner and Hardware Orchestration Installer
Automated 1-Click download and atomic deployment of precompiled ARM64 Bionic binaries,
shared libraries, and compute shaders across Termux and Android edge environments.
"""

import hashlib
import logging
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ._version import __version__

logger = logging.getLogger("ameva_runtime.installer")

AMEVA_RUNTIME_GITHUB_REPO = "uno-km/ameva-runtime"
GITHUB_RELEASE_LATEST = f"https://github.com/{AMEVA_RUNTIME_GITHUB_REPO}/releases/latest/download"


def get_release_base_url() -> str:
    """Resolve GitHub release asset base URL dynamically via 3-tier fallback chain:
    1. Explicit environment variable: AMEVA_RELEASE_BASE or AMEVA_RELEASE_TAG
    2. Versioned release tag matching current package version: v{__version__}
    3. Latest release download endpoint: GITHUB_RELEASE_LATEST
    """
    if custom_base := os.environ.get("AMEVA_RELEASE_BASE"):
        return custom_base.rstrip("/")
    if custom_tag := os.environ.get("AMEVA_RELEASE_TAG"):
        tag = custom_tag if custom_tag.startswith("v") else f"v{custom_tag}"
        return f"https://github.com/{AMEVA_RUNTIME_GITHUB_REPO}/releases/download/{tag}"
    return f"https://github.com/{AMEVA_RUNTIME_GITHUB_REPO}/releases/download/v{__version__}"


GITHUB_RELEASE_BASE = get_release_base_url()

# Standard filesystem locations in Termux / Linux
HOME = Path(os.environ.get("HOME", os.path.expanduser("~")))
PREFIX = Path(os.environ.get("PREFIX", "/data/data/com.termux/files/usr"))
LOCAL_BIN = HOME / ".local" / "bin"
LOCAL_LIB = HOME / ".local" / "lib"
PREFIX_BIN = PREFIX / "bin"
PREFIX_LIB = PREFIX / "lib"
AMEVA_SHARE = HOME / ".local" / "share" / "ameva"
AMEVA_SHADERS = AMEVA_SHARE / "shaders"
AMEVA_MODELS = HOME / ".cache" / "ameva" / "models"


MAX_ARCHIVE_FILE_COUNT = 1000
MAX_ARCHIVE_SINGLE_FILE_SIZE = 250 * 1024 * 1024  # 250 MB
MAX_ARCHIVE_TOTAL_EXPANDED_SIZE = 500 * 1024 * 1024  # 500 MB
MAX_ARCHIVE_PATH_DEPTH = 16


def safe_extract_tar(
    archive_path: Path,
    target_dir: Path,
    max_file_count: int = MAX_ARCHIVE_FILE_COUNT,
    max_single_file_size: int = MAX_ARCHIVE_SINGLE_FILE_SIZE,
    max_total_size: int = MAX_ARCHIVE_TOTAL_EXPANDED_SIZE,
    max_path_depth: int = MAX_ARCHIVE_PATH_DEPTH,
) -> None:
    """Safely extracts a tar.gz archive ensuring no member escapes target_dir,
    strictly permitting only regular files and directories, rejecting special members/links,
    normalizing paths with case-folding on Windows, rejecting negative sizes,
    and enforcing member count, file size, total expanded size, and path depth limits.
    All members are pre-validated before any file extraction begins.
    """
    if target_dir.is_symlink():
        raise RuntimeError(f"Target directory cannot be a symbolic link: {target_dir}")

    root = target_dir.resolve()
    seen_paths = set()
    total_expanded_size = 0
    member_count = 0
    validated_members = []

    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_count += 1
            if member_count > max_file_count:
                raise RuntimeError(
                    f"MAX_FILE_COUNT_ENFORCED: archive member count ({member_count}) exceeds limit ({max_file_count})"
                )

            # Strictly allow only regular files and directories
            if member.islnk() or member.issym():
                raise RuntimeError(
                    f"Archive links are not permitted: {member.name}"
                )
            if member.isfifo():
                raise RuntimeError(
                    f"FIFO_REJECTED: FIFO special member rejected: {member.name}"
                )
            if member.ischr():
                raise RuntimeError(
                    f"CHAR_DEVICE_REJECTED: character device special member rejected: {member.name}"
                )
            if member.isblk():
                raise RuntimeError(
                    f"BLOCK_DEVICE_REJECTED: block device special member rejected: {member.name}"
                )
            if not (member.isdir() or member.isreg()):
                raise RuntimeError(
                    f"SPECIAL_MEMBER_REJECTED: unsupported archive member type: {member.name}"
                )

            # Prevent negative or invalid integer sizes
            if member.size < 0 or member.size > 2**63 - 1:
                raise RuntimeError(
                    f"INVALID_MEMBER_SIZE: member {member.name} has invalid size {member.size}"
                )

            destination = (root / member.name).resolve()
            try:
                rel = destination.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Archive path escapes target directory: {member.name}"
                ) from exc

            depth = len(rel.parts)
            if depth > max_path_depth:
                raise RuntimeError(
                    f"MAX_PATH_DEPTH_ENFORCED: path depth {depth} exceeds limit {max_path_depth}: {member.name}"
                )

            # POSIX as_posix() normalization + case-insensitivity on Windows
            posix_path = rel.as_posix()
            norm_rel = posix_path.lower() if sys.platform == "win32" else posix_path
            if norm_rel in seen_paths:
                raise RuntimeError(
                    f"DUPLICATE_PATH_REJECTED: duplicate path detected in archive: {member.name}"
                )
            seen_paths.add(norm_rel)

            if member.isreg():
                if member.size > max_single_file_size:
                    raise RuntimeError(
                        f"MAX_SINGLE_FILE_SIZE_ENFORCED: member {member.name} size ({member.size} bytes) exceeds limit ({max_single_file_size} bytes)"
                    )
                total_expanded_size += member.size
                if total_expanded_size > max_total_size:
                    raise RuntimeError(
                        f"MAX_TOTAL_EXPANDED_SIZE_ENFORCED: total expanded size ({total_expanded_size} bytes) exceeds limit ({max_total_size} bytes)"
                    )

            validated_members.append(member)

        # Pre-validation complete: extract only explicitly validated members
        tar.extractall(path=target_dir, members=validated_members)


def verify_sha256(path: Path, expected: Optional[str]) -> None:
    """Verifies the SHA-256 digest of a downloaded file against an expected checksum.
    Enforces that expected hash must exist and match exactly (P1 Checksum Enforcement).
    If verification fails, the file is immediately unlinked.
    """
    if not expected:
        path.unlink(missing_ok=True)
        raise RuntimeError("Missing expected SHA-256 checksum: unauthenticated asset rejected")
    normalized = expected.strip().lower()
    if len(normalized) != 64 or not all(c in "0123456789abcdef" for c in normalized):
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Invalid expected SHA-256 digest format: {expected}")

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != normalized:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {path.name}: expected={normalized}, actual={actual}"
        )


@dataclass
class AssetSpec:
    name: str
    filename: str
    target_path: Path
    is_archive: bool = False
    archive_extract_target: Optional[str] = None
    is_executable: bool = True
    sha256: Optional[str] = None
    legacy_symlinks: List[Path] = field(default_factory=list)
    description: str = ""


# SSOT Registry of Native Compiled Assets
NATIVE_ASSETS: Dict[str, AssetSpec] = {
    "diffusion": AssetSpec(
        name="diffusion",
        filename="sd-cli-vulkan-android-arm64.tar.gz",
        target_path=LOCAL_BIN / "sd-cli",
        is_archive=True,
        archive_extract_target="sd-cli",
        is_executable=True,
        legacy_symlinks=[
            HOME / ".cache" / "termux-diffusion" / "bin" / "sd-cli",
            HOME / ".cache" / "termux-diffusion" / "bin" / "sd-cli-vulkan",
            PREFIX_BIN / "sd-cli",
        ],
        description="Stable Diffusion On-Device Vulkan CLI Engine (sd-cli)",
    ),
    "stt": AssetSpec(
        name="stt",
        filename="whisper-cli-android-arm64",
        target_path=LOCAL_BIN / "whisper-cli",
        is_archive=False,
        is_executable=True,
        legacy_symlinks=[
            PREFIX_BIN / "whisper-cli",
            LOCAL_BIN / "whisper-cpp",
        ],
        description="Whisper Speech-to-Text Native Engine (whisper-cli)",
    ),
    "tts": AssetSpec(
        name="tts",
        filename="sherpa-ncnn-offline-tts-vulkan-arm64.tar.gz",
        target_path=LOCAL_BIN / "sherpa-ncnn-offline-tts",
        is_archive=True,
        archive_extract_target="sherpa-ncnn-offline-tts",
        is_executable=True,
        legacy_symlinks=[
            HOME / "sherpa-ncnn" / "build-vulkan" / "bin" / "sherpa-ncnn-offline-tts",
            PREFIX_BIN / "sherpa-ncnn-offline-tts",
        ],
        description="Sherpa-NCNN Vulkan Neural Speech Synthesis (sherpa-ncnn-offline-tts)",
    ),
    "libomp": AssetSpec(
        name="libomp",
        filename="libomp-android-arm64.so",
        target_path=PREFIX_LIB / "libomp.so",
        is_archive=False,
        is_executable=False,
        legacy_symlinks=[
            LOCAL_LIB / "libomp.so",
        ],
        description="High-Performance OpenMP Multi-Threading Runtime (libomp.so)",
    ),
    "libegl_shim": AssetSpec(
        name="libegl_shim",
        filename="libegl_shim-android-arm64.so",
        target_path=PREFIX_LIB / "libegl_shim.so",
        is_archive=False,
        is_executable=False,
        legacy_symlinks=[
            LOCAL_LIB / "libegl_shim.so",
        ],
        description="Termux Headless Vulkan/EGL Bionic Shim Driver (libegl_shim.so)",
    ),
    "matmul_spv": AssetSpec(
        name="matmul_spv",
        filename="matmul-tensor-compute.spv",
        target_path=AMEVA_SHADERS / "matmul.spv",
        is_archive=False,
        is_executable=False,
        legacy_symlinks=[
            PREFIX / "share" / "ameva" / "shaders" / "matmul.spv",
        ],
        description="AMEVA Universal Mobile High-Performance MatMul SPIR-V Kernel (matmul-tensor-compute.spv)",
    ),
}


class NativeAssetManager:
    """1-Click Native Hardware Asset Provisioner for AMEVA Runtime."""

    def __init__(self, base_url: Optional[str] = None, force: bool = False):
        self.base_url = (base_url or get_release_base_url()).rstrip("/")
        self.force = force

    def provision_all(self, modalities: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Provisions native assets across all or specified modalities.
        Creates missing directories, downloads binaries/libraries, enforces executable permissions,
        and establishes dual-track compatibility symlinks.
        """
        targets = modalities or list(NATIVE_ASSETS.keys())
        results: Dict[str, bool] = {}

        print("=" * 72)
        print(f"   AMEVA Runtime: Native Asset Auto-Provisioner (v{__version__})")
        print("=" * 72)
        print(f"[*] Base Distribution Server : {self.base_url}")
        print(f"[*] Target System Root       : {HOME}")
        print(f"[*] Force Overwrite Policy   : {self.force}\n")

        for key in targets:
            if key not in NATIVE_ASSETS:
                logger.warning("Unknown asset spec: %s", key)
                continue
            spec = NATIVE_ASSETS[key]
            success = self.provision_asset(spec)
            results[key] = success

        self._check_path_environment()
        print("=" * 72)
        passed = sum(1 for v in results.values() if v)
        print(f"[RESULT] Provisioned {passed}/{len(results)} native assets successfully.")
        print("=" * 72)
        return results

    def provision_asset(self, spec: AssetSpec) -> bool:
        """Provisions a single asset with checksum verification and dual-track symlinks."""
        print(f"[*] [{spec.name.upper()}] {spec.description}...")
        target_dir = spec.target_path.parent
        # User Request 2: Create directory if it does not exist
        target_dir.mkdir(parents=True, exist_ok=True)

        # User Request 1: Check existing vs force overwrite
        if spec.target_path.exists() and not self.force:
            print(f"    -> Existing binary verified at {spec.target_path} (use --force to overwrite)")
            self._create_dual_track_links(spec)
            return True

        # Check if expected sha256 is provided either on spec or via env
        expected_sha256 = spec.sha256 or os.environ.get(f"AMEVA_SHA256_{spec.name.upper()}")
        if not expected_sha256:
            print(f"    -> [FAIL] Missing required SHA-256 checksum for {spec.name}. Supply chain policy strictly requires pinned digest.")
            return False

        download_url = f"{self.base_url}/{spec.filename}"
        temp_dest = target_dir / f".{spec.filename}.tmp"
        extract_tmp_dir = target_dir / f".tmp_extract_{spec.name}"

        try:
            # 1. Download with dynamic 3-tier fallback (tag -> latest)
            print(f"    -> Downloading: {spec.filename} ...")
            req = urllib.request.Request(download_url, headers={"User-Agent": f"AMEVA-Installer/{__version__}"})
            try:
                resp = urllib.request.urlopen(req, timeout=60)
            except urllib.error.HTTPError as http_err:
                if http_err.code == 404 and self.base_url != GITHUB_RELEASE_LATEST:
                    fallback_url = f"{GITHUB_RELEASE_LATEST}/{spec.filename}"
                    print(f"    -> Release tag URL returned 404; falling back to latest release: {fallback_url}")
                    fallback_req = urllib.request.Request(fallback_url, headers={"User-Agent": f"AMEVA-Installer/{__version__}"})
                    resp = urllib.request.urlopen(fallback_req, timeout=60)
                else:
                    raise
            with resp, open(temp_dest, "wb") as out_f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)

            # 2. Cryptographic SHA-256 verification BEFORE any extraction or deployment
            verify_sha256(temp_dest, expected_sha256)
            print(f"    -> [PASS] Cryptographically verified SHA-256 checksum ({expected_sha256[:16]}...)")

            # 3. Extract or Deploy via safe paths
            if spec.is_archive:
                print(f"    -> Safely unpacking archive for {spec.name}...")
                if extract_tmp_dir.exists():
                    shutil.rmtree(extract_tmp_dir, ignore_errors=True)
                extract_tmp_dir.mkdir(parents=True, exist_ok=True)
                try:
                    safe_extract_tar(temp_dest, extract_tmp_dir)
                    if spec.archive_extract_target:
                        target_candidates = list(extract_tmp_dir.rglob(spec.archive_extract_target))
                        if not target_candidates:
                            raise RuntimeError(f"Archive member {spec.archive_extract_target} not found in {spec.filename}")
                        extracted_file = target_candidates[0]
                        if spec.target_path.exists():
                            spec.target_path.unlink()
                        shutil.move(str(extracted_file), str(spec.target_path))
                    else:
                        for item in extract_tmp_dir.iterdir():
                            dest_item = target_dir / item.name
                            if dest_item.exists():
                                if dest_item.is_dir():
                                    shutil.rmtree(dest_item)
                                else:
                                    dest_item.unlink()
                            shutil.move(str(item), str(dest_item))
                finally:
                    if extract_tmp_dir.exists():
                        shutil.rmtree(extract_tmp_dir, ignore_errors=True)
            else:
                if spec.target_path.exists():
                    spec.target_path.unlink()
                shutil.move(str(temp_dest), str(spec.target_path))

            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)

            # 4. Set Executable Mode AFTER verification and extraction
            if spec.is_executable and os.name != "nt":
                spec.target_path.chmod(0o755)

            print(f"    -> [PASS] Installed at {spec.target_path}")

            # 5. Create Dual-Track Compatibility Links
            self._create_dual_track_links(spec)
            return True

        except Exception as e:
            print(f"    -> [FAIL] Error provisioning {spec.name}: {e}")
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
            if extract_tmp_dir.exists():
                shutil.rmtree(extract_tmp_dir, ignore_errors=True)
            return False

    def _create_dual_track_links(self, spec: AssetSpec) -> None:
        """Creates backward-compatibility symlinks or hard copies for legacy toolchains."""
        if not spec.target_path.exists():
            return

        for legacy_path in spec.legacy_symlinks:
            try:
                legacy_dir = legacy_path.parent
                legacy_dir.mkdir(parents=True, exist_ok=True)

                if legacy_path.exists() or legacy_path.is_symlink():
                    if not self.force:
                        continue
                    try:
                        legacy_path.unlink()
                    except Exception:
                        pass

                # Attempt symbolic link first, fallback to copy
                try:
                    if os.name != "nt":
                        os.symlink(str(spec.target_path), str(legacy_path))
                    else:
                        shutil.copy2(str(spec.target_path), str(legacy_path))
                except (OSError, NotImplementedError):
                    shutil.copy2(str(spec.target_path), str(legacy_path))

                logger.debug("Dual-track bridge established: %s -> %s", legacy_path, spec.target_path)
            except Exception as e:
                logger.debug("Failed to establish legacy link at %s: %s", legacy_path, e)

    def _check_path_environment(self) -> None:
        """Notifies user if ~/.local/bin is absent from current PATH."""
        cur_path = os.environ.get("PATH", "")
        local_bin_str = str(LOCAL_BIN)
        if local_bin_str not in cur_path and os.name != "nt":
            print("\n[NOTICE] ~/.local/bin is not in your current PATH.")
            print("  To make all installed tools globally accessible, run:")
            print('  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc && source ~/.bashrc')


def provision_native_assets(
    modalities: Optional[List[str]] = None,
    force: bool = False,
    base_url: Optional[str] = None,
) -> Dict[str, bool]:
    """Programmatic entrypoint for AMEVA Native Asset Auto-Provisioning."""
    manager = NativeAssetManager(base_url=base_url, force=force)
    return manager.provision_all(modalities=modalities)
