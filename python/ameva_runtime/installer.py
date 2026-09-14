"""
AMEVA Native Asset Provisioner and Hardware Orchestration Installer
Automated 1-Click download and atomic deployment of precompiled ARM64 Bionic binaries,
shared libraries, and compute shaders across Termux and Android edge environments.
"""

import datetime
import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
LOCAL_BIN = HOME / ".local" / "bin"
LOCAL_LIB = HOME / ".local" / "lib"
AMEVA_SHARE = HOME / ".local" / "share" / "ameva"
AMEVA_MANIFESTS = AMEVA_SHARE / "manifests"
AMEVA_RELEASES = AMEVA_SHARE / "releases"
AMEVA_CURRENT = AMEVA_SHARE / "current"
AMEVA_LOCKS = AMEVA_SHARE / "locks"
AMEVA_MODELS = HOME / ".cache" / "ameva" / "models"

try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False


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


def _calc_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest of a file in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def verify_sha256(path: Path, expected: str) -> None:
    """Verifies the SHA-256 digest of a target file against an expected checksum.
    Strictly verifies hash format and digest equality.
    Does NOT mutate or delete the file under test (Lifecycle Isolation).
    """
    if not expected:
        raise RuntimeError("Missing expected SHA-256 checksum: unauthenticated asset rejected")
    normalized = expected.strip().lower()
    if len(normalized) != 64 or not all(c in "0123456789abcdef" for c in normalized):
        raise RuntimeError(f"Invalid expected SHA-256 digest format: {expected}")

    actual = _calc_file_sha256(path)
    if actual != normalized:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}: expected={normalized}, actual={actual}"
        )


def _verify_file_format(target: Path, is_executable: bool = False) -> None:
    """Verifies binary format: ELF64 little-endian AArch64 ABI, ET_DYN shared library, or SPIR-V."""
    if not target.exists():
        raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Target path does not exist: {target}")

    # 1. SPIR-V binary verification (magic: 0x07230203 -> little-endian bytes: b"\x03\x02\x23\x07")
    if target.suffix == ".spv":
        with target.open("rb") as stream:
            magic = stream.read(4)
        if magic != b"\x03\x02\x23\x07":
            raise RuntimeError(
                f"DEPLOYMENT_VERIFICATION_FAILED: Invalid SPIR-V binary (magic mismatch): {target}"
            )
        return

    # 2. Executable permission check on POSIX
    if is_executable and os.name != "nt":
        if not os.access(str(target), os.X_OK):
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Target file lacks execute permission: {target}")

    # 3. ELF binary verification (Executable or Shared Library)
    if is_executable or target.suffix == ".so" or ".so." in target.name:
        with target.open("rb") as stream:
            header = stream.read(64)
        if len(header) < 20:
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: ELF header truncated: {target}")
        if header[:4] != b"\x7fELF":
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Native asset is not ELF: {target}")
        if header[4] != 2:  # ELFCLASS64
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Expected ELF64: {target}")
        if header[5] != 1:  # ELFDATA2LSB (little-endian)
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Expected little-endian AArch64 ELF: {target}")
        if header[6] != 1:  # EV_CURRENT
            raise RuntimeError(f"DEPLOYMENT_VERIFICATION_FAILED: Unsupported ELF version ({header[6]}): {target}")

        e_type = int.from_bytes(header[16:18], "little")
        machine = int.from_bytes(header[18:20], "little")

        if machine != 183:  # 183 is EM_AARCH64
            raise RuntimeError(
                f"DEPLOYMENT_VERIFICATION_FAILED: Architecture mismatch for {target} "
                f"(machine={machine}, expected AArch64/183)"
            )

        # Shared library (.so) must be ET_DYN (3)
        if (target.suffix == ".so" or ".so." in target.name) and e_type != 3:
            raise RuntimeError(
                f"DEPLOYMENT_VERIFICATION_FAILED: Shared library {target} is not ET_DYN (e_type={e_type}, expected 3)"
            )


MAX_DOWNLOAD_SIZE = 600 * 1024 * 1024  # 600 MB


@dataclass(frozen=True)
class AssetSpec:
    name: str
    filename: str
    download_sha256: str
    binary_relpath: str
    canonical_name: str
    description: str = ""


# Streamlined SSOT Registry: Engine Bundles Only (No orphan libraries or shader downloads)
NATIVE_ASSETS: Dict[str, AssetSpec] = {
    "stt": AssetSpec(
        name="stt",
        filename="whisper-cli-vulkan-android-arm64.tar.gz",
        download_sha256="90a2f4fd275aa13012e95f3fae5b00c2abc5079a50d2997ffb227355e6b6c944",
        binary_relpath="bin/whisper-cli",
        canonical_name="whisper-cli",
        description="Whisper Speech-to-Text Native Engine with Vulkan GPU Acceleration",
    ),
    "tts": AssetSpec(
        name="tts",
        filename="sherpa-ncnn-offline-tts-vulkan-arm64.tar.gz",
        download_sha256="c112a4de96b31cee2d92b610fcc350ba4c2b2331eee25f3f86520b200992e8d5",
        binary_relpath="sherpa-ncnn-offline-tts",
        canonical_name="sherpa-ncnn-offline-tts",
        description="Sherpa-NCNN Vulkan Neural Speech Synthesis",
    ),
    "diffusion": AssetSpec(
        name="diffusion",
        filename="sd-cli-vulkan-android-arm64.tar.gz",
        download_sha256="0340cf0a8ca4172f184a9b5b9e419905e35b51a00e07e88841d8537224235fbc",
        binary_relpath="sd-cli-vulkan",
        canonical_name="sd-cli",
        description="Stable Diffusion On-Device Vulkan CLI Engine",
    ),
}


class NativeAssetManager:
    """Zero-Silent-Fallback Native Hardware Asset Provisioner for AMEVA Runtime."""

    def __init__(self, base_url: Optional[str] = None, force: bool = False, force_repair: bool = False):
        self.base_url = (base_url or get_release_base_url()).rstrip("/")
        self.force = force
        self.force_repair = force_repair

    def provision_all(self, modalities: Optional[List[str]] = None) -> Dict[str, Any]:
        """Provisions native assets across all or specified engine modalities."""
        targets = modalities or list(NATIVE_ASSETS.keys())
        results: Dict[str, Any] = {}

        print("=" * 72)
        print(f"   AMEVA Runtime: Native Engine Bundle Provisioner (v{__version__})")
        print("=" * 72)
        print(f"[*] Base Distribution Server : {self.base_url}")
        print(f"[*] Target System Root       : {HOME}")
        print(f"[*] Force Overwrite Policy   : {self.force}\n")

        for key in targets:
            if key not in NATIVE_ASSETS:
                raise KeyError(f"Unknown native engine bundle: '{key}'. Available: {list(NATIVE_ASSETS.keys())}")
            spec = NATIVE_ASSETS[key]
            if not spec.download_sha256 or len(spec.download_sha256) != 64:
                raise RuntimeError(f"Registry integrity error: download_sha256 missing or invalid for bundle '{key}'")

        for key in targets:
            spec = NATIVE_ASSETS[key]
            res = self.provision_asset(spec)
            results[key] = res

        self._check_path_environment()
        print("=" * 72)
        print(f"[RESULT] All requested bundles individually deployed and verified (ASSET_DEPLOYED_VERIFIED).")
        print("=" * 72)
        return results

    def _verify_existing_bundle_or_raise(self, spec: AssetSpec) -> None:
        """Cryptographically verifies that an existing bundle installation is intact.
        Under Zero-Silent-Fallback policy, any tampering, file corruption, or link mismatch
        MUST raise an explicit exception and fail fast; it is NEVER silently re-downloaded.
        """
        current_link = AMEVA_CURRENT / spec.name
        canonical_bin = LOCAL_BIN / spec.canonical_name

        if not canonical_bin.exists():
            raise RuntimeError(
                f"Broken installation: Canonical executable missing at {canonical_bin} for {spec.name}. "
                f"Run with --force-repair to re-install."
            )

        if not current_link.exists():
            raise RuntimeError(
                f"Broken installation: Current release link missing at {current_link} for {spec.name}. "
                f"Run with --force-repair to re-install."
            )

        # P0-5: Verify canonical binary resolves strictly to expected target in current release
        try:
            expected_binary = (current_link / spec.binary_relpath).resolve(strict=True)
            actual_binary = canonical_bin.resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError) as e:
            raise RuntimeError(
                f"Broken installation: Failed resolving binary targets for {spec.name}: {e}. "
                f"Run with --force-repair to re-install."
            )

        if actual_binary != expected_binary:
            raise RuntimeError(
                f"Deployment tampering detected: Canonical binary target mismatch for {spec.name}. "
                f"actual={actual_binary}, expected={expected_binary}. Refusing silent re-download. "
                f"Run with --force-repair to authorize re-provisioning."
            )

        # P0-4: Locate and read manifest.json
        manifest_path = current_link / "manifest.json"
        if not manifest_path.is_file():
            manifest_path = AMEVA_MANIFESTS / f"{spec.name}.json"
        if not manifest_path.is_file():
            raise RuntimeError(
                f"Tampering/integrity violation: Missing cryptographic manifest for installed bundle '{spec.name}' "
                f"at {manifest_path}. Untracked or corrupt installation detected. "
                f"Run with --force-repair to authorize re-provisioning."
            )

        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest_data = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Corrupt manifest file {manifest_path} for {spec.name}: {exc}. "
                f"Run with --force-repair to authorize re-provisioning."
            ) from exc

        # Validate bundle metadata
        if manifest_data.get("bundle_id") != spec.name:
            raise RuntimeError(
                f"Manifest metadata violation: bundle_id mismatch ({manifest_data.get('bundle_id')} != {spec.name})"
            )
        if manifest_data.get("download_sha256") != spec.download_sha256:
            raise RuntimeError(
                f"Manifest metadata violation: download_sha256 mismatch for {spec.name} "
                f"({manifest_data.get('download_sha256')} != {spec.download_sha256})"
            )
        if manifest_data.get("target_architecture") != "aarch64":
            raise RuntimeError(
                f"Manifest metadata violation: architecture mismatch ({manifest_data.get('target_architecture')} != aarch64)"
            )

        # Validate primary binary format and SHA-256
        expected_bin_sha = manifest_data.get("binary_sha256", "").lower()
        if not expected_bin_sha:
            raise RuntimeError(f"Corrupt manifest: missing binary_sha256 for {spec.name}")
        actual_bin_sha = _calc_file_sha256(actual_binary)
        if actual_bin_sha != expected_bin_sha:
            raise RuntimeError(
                f"Cryptographic hash mismatch for primary binary '{actual_binary}': "
                f"actual={actual_bin_sha}, expected={expected_bin_sha}. Tampering detected. "
                f"Run with --force-repair to authorize re-provisioning."
            )

        _verify_file_format(actual_binary, is_executable=True)

        # Validate all deployed files in manifest (existence, size, SHA-256, format)
        deployed_files = manifest_data.get("deployed_files", {})
        if not deployed_files:
            raise RuntimeError(f"Corrupt manifest: deployed_files is empty for {spec.name}")

        for rel_path, entry in deployed_files.items():
            file_path = (current_link / rel_path).resolve(strict=False)
            if not file_path.is_file():
                raise RuntimeError(
                    f"Integrity violation: Deployed file missing: '{file_path}' (from manifest entry '{rel_path}'). "
                    f"Run with --force-repair to authorize re-provisioning."
                )

            expected_size = entry.get("size")
            if expected_size is not None and file_path.stat().st_size != expected_size:
                raise RuntimeError(
                    f"Integrity violation: File size mismatch for '{file_path}': "
                    f"expected={expected_size}, actual={file_path.stat().st_size}. Tampering detected."
                )

            expected_sha = entry.get("sha256", "").lower()
            if not expected_sha:
                raise RuntimeError(f"Corrupt manifest: missing sha256 entry for '{rel_path}'")

            actual_sha = _calc_file_sha256(file_path)
            if actual_sha != expected_sha:
                raise RuntimeError(
                    f"Integrity violation: Cryptographic hash mismatch for '{file_path}': "
                    f"expected={expected_sha}, actual={actual_sha}. Tampering detected. "
                    f"Run with --force-repair to authorize re-provisioning."
                )

            if file_path.suffix == ".so" or ".so." in file_path.name:
                _verify_file_format(file_path, is_executable=False)

    def _generate_and_save_bundle_manifest(
        self, spec: AssetSpec, release_id: str, release_dir: Path
    ) -> Path:
        """Generates and writes an authenticated cryptographic manifest file for the provisioned engine bundle."""
        AMEVA_MANIFESTS.mkdir(parents=True, exist_ok=True)
        manifest_path = AMEVA_MANIFESTS / f"{spec.name}.json"

        deployed_entries = {}
        required_libs = []

        primary_bin = release_dir / spec.binary_relpath
        bin_sha = _calc_file_sha256(primary_bin) if primary_bin.is_file() else ""

        for f in release_dir.rglob("*"):
            if f.is_file() and not f.is_symlink():
                rel = f.relative_to(release_dir).as_posix()
                file_sha = _calc_file_sha256(f)
                deployed_entries[rel] = {
                    "path": str(f),
                    "sha256": file_sha,
                    "size": f.stat().st_size,
                }
                if f.suffix == ".so" or ".so." in f.name:
                    required_libs.append(f.name)

        manifest_data = {
            "bundle_id": spec.name,
            "version": __version__,
            "release_id": release_id,
            "archive_filename": spec.filename,
            "download_sha256": spec.download_sha256,
            "target_architecture": "aarch64",
            "compute_abi": "elf64-littleaarch64",
            "backend_feature": "vulkan" if "vulkan" in spec.filename else "cpu",
            "binary_relpath": spec.binary_relpath,
            "canonical_name": spec.canonical_name,
            "binary_path": str(LOCAL_BIN / spec.canonical_name),
            "binary_sha256": bin_sha,
            "deployed_files": deployed_entries,
            "required_shared_libraries": sorted(list(set(required_libs))),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        manifest_content = json.dumps(manifest_data, indent=2)
        staged_manifest = manifest_path.with_name(f".{manifest_path.name}.staged")
        with open(staged_manifest, "w", encoding="utf-8") as f:
            f.write(manifest_content)
        os.replace(staged_manifest, manifest_path)

        rel_manifest = release_dir / "manifest.json"
        staged_rel = rel_manifest.with_name(".manifest.json.staged")
        with open(staged_rel, "w", encoding="utf-8") as f:
            f.write(manifest_content)
        os.replace(staged_rel, rel_manifest)

        return manifest_path

    def provision_asset(self, spec: AssetSpec) -> Dict[str, Any]:
        """Provisions a single engine bundle with file lock, release directory swap, and fail-fast."""
        AMEVA_LOCKS.mkdir(parents=True, exist_ok=True)
        lock_file = AMEVA_LOCKS / f"{spec.name}.lock"

        if HAS_FCNTL:
            with lock_file.open("a+") as lock_stream:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
                try:
                    return self._provision_asset_locked(spec)
                finally:
                    fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        else:
            if os.name != "nt":
                raise RuntimeError("Installation lock unavailable: fcntl is required on POSIX / Termux")
            return self._provision_asset_locked(spec)

    def _provision_asset_locked(self, spec: AssetSpec) -> Any:
        """Internal locked transaction for provisioning an engine bundle."""
        import tempfile

        if not spec.download_sha256:
            raise RuntimeError(f"Missing pinned SHA-256 for asset: {spec.name}")
        expected_sha = spec.download_sha256

        print(f"[*] [{spec.name.upper()}] {spec.description}...")
        canonical_bin = LOCAL_BIN / spec.canonical_name
        current_link = AMEVA_CURRENT / spec.name
        LOCAL_BIN.mkdir(parents=True, exist_ok=True)

        is_installed = current_link.exists() or canonical_bin.exists()

        # Strict Zero-Silent-Fallback: If installed, verify or raise. Never silently re-download!
        if is_installed and not (self.force or self.force_repair):
            self._verify_existing_bundle_or_raise(spec)
            print(f"    -> [PASS] Existing engine bundle verified: {canonical_bin}")
            manifest_path = AMEVA_MANIFESTS / f"{spec.name}.json"
            return {
                "status": "ASSET_DEPLOYED_VERIFIED",
                "name": spec.name,
                "target_path": str(canonical_bin),
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "reused_existing": True,
            }

        if is_installed and (self.force or self.force_repair):
            print(f"    -> [NOTICE] Existing bundle present but repair requested (--force-repair). Proceeding with clean re-provisioning.")

        download_url = f"{self.base_url}/{spec.filename}"

        with tempfile.TemporaryDirectory(prefix=f".pm-tmp-{spec.name}-", dir=str(LOCAL_BIN)) as temp_work_dir:
            temp_dest = Path(temp_work_dir) / spec.filename
            extract_tmp_dir = Path(temp_work_dir) / "extract"

            # 1. Direct Download with size bounding: strictly reject 404 silent fallback
            print(f"    -> Downloading: {spec.filename} ...")
            req = urllib.request.Request(download_url, headers={"User-Agent": f"AMEVA-Installer/{__version__}"})
            try:
                resp = urllib.request.urlopen(req, timeout=60)
            except urllib.error.HTTPError as http_err:
                raise RuntimeError(
                    f"Release bundle unavailable: url={download_url}, status={http_err.code}"
                ) from http_err

            downloaded = 0
            with resp, open(temp_dest, "wb") as out_f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > MAX_DOWNLOAD_SIZE:
                        raise RuntimeError(
                            f"Download exceeds maximum size limit: {downloaded} > {MAX_DOWNLOAD_SIZE}"
                        )
                    out_f.write(chunk)

            # 2. Cryptographic SHA-256 verification of archive BEFORE any extraction
            verify_sha256(temp_dest, expected_sha)
            print(f"    -> [PASS] Cryptographically verified bundle SHA-256 ({expected_sha[:16]}...)")

            # 3. Unpack into extract_tmp_dir
            print(f"    -> Safely unpacking engine bundle for {spec.name}...")
            extract_tmp_dir.mkdir(parents=True, exist_ok=True)
            safe_extract_tar(temp_dest, extract_tmp_dir)

            # Verify primary executable existence & format
            primary_bin = extract_tmp_dir / spec.binary_relpath
            if not primary_bin.is_file():
                raise RuntimeError(
                    f"Bundle integrity error: Primary executable '{spec.binary_relpath}' not found in {spec.filename}"
                )
            _verify_file_format(primary_bin, is_executable=True)

            # Verify any bundled shared libraries
            for so_file in extract_tmp_dir.rglob("*.so*"):
                if so_file.is_file() and not so_file.is_symlink():
                    _verify_file_format(so_file, is_executable=False)

            # 4. Stage Release Directory: immutable releases are never modified or replaced
            release_root = AMEVA_RELEASES / spec.name
            if self.force or self.force_repair:
                timestamp = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
                release_id = f"v{__version__}-{spec.download_sha256[:12]}-repair-{timestamp}"
            else:
                release_id = f"v{__version__}-{spec.download_sha256[:12]}"

            release_dir = release_root / release_id
            staging_release = release_root / f".{release_id}.staging"

            if release_dir.exists():
                raise RuntimeError(
                    f"Immutable release directory already exists: {release_dir}. "
                    f"Existing releases cannot be modified or replaced."
                )

            if staging_release.exists():
                shutil.rmtree(staging_release)
            staging_release.parent.mkdir(parents=True, exist_ok=True)

            shutil.copytree(str(extract_tmp_dir), str(staging_release))

            # Mark primary binary executable
            staged_primary = staging_release / spec.binary_relpath
            if os.name != "nt":
                staged_primary.chmod(0o755)

            # Move staging release directly into final release_dir
            # (No existing release directory is EVER touched, moved, or deleted)
            os.replace(staging_release, release_dir)

            # 5. Generate and save manifest inside release_dir and canonical manifests/
            manifest_path = self._generate_and_save_bundle_manifest(spec, release_id, release_dir)

            # 6. Atomically swap current symlink: ~/.local/share/ameva/current/<name> -> release_dir
            AMEVA_CURRENT.mkdir(parents=True, exist_ok=True)
            current_symlink = AMEVA_CURRENT / spec.name
            tmp_current = current_symlink.parent / f".tmp_{spec.name}_{release_id}"
            if tmp_current.exists() or tmp_current.is_symlink():
                tmp_current.unlink(missing_ok=True)
            if os.name != "nt":
                tmp_current.symlink_to(release_dir)
                os.replace(tmp_current, current_symlink)

            # 7. Atomically link primary executable into ~/.local/bin/<canonical_name>
            tmp_bin = canonical_bin.with_name(f".tmp_{canonical_bin.name}")
            if tmp_bin.exists() or tmp_bin.is_symlink():
                tmp_bin.unlink(missing_ok=True)

            target_target = (current_symlink / spec.binary_relpath) if os.name != "nt" else (release_dir / spec.binary_relpath)
            if os.name != "nt":
                tmp_bin.symlink_to(target_target)
                os.replace(tmp_bin, canonical_bin)
            else:
                shutil.copy2(str(release_dir / spec.binary_relpath), str(tmp_bin))
                os.replace(tmp_bin, canonical_bin)

            # Final verification of canonical binary
            _verify_file_format(canonical_bin, is_executable=True)

            print(f"    -> [PASS] Installed and deployment-verified: {canonical_bin}")

            return {
                "status": "ASSET_DEPLOYED_VERIFIED",
                "name": spec.name,
                "target_path": str(canonical_bin),
                "download_sha256": spec.download_sha256,
                "manifest_path": str(manifest_path),
                "reused_existing": False,
            }

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
    force_repair: bool = False,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Programmatic entrypoint for AMEVA Native Asset Auto-Provisioning."""
    manager = NativeAssetManager(base_url=base_url, force=force, force_repair=force_repair)
    return manager.provision_all(modalities=modalities)


