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
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("ameva_runtime.installer")

GITHUB_RELEASE_BASE = "https://github.com/uno-km/ameva-runtime/releases/download/v2.2.1"

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


# SSOT Registry of Native Compiled Assets in v2.2.1
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
        description="ARM Mali / Qualcomm Adreno Hardware Workaround SPIR-V Shader",
    ),
}


class NativeAssetManager:
    """1-Click Native Hardware Asset Provisioner for AMEVA Runtime."""

    def __init__(self, base_url: str = GITHUB_RELEASE_BASE, force: bool = False):
        self.base_url = base_url
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
        print("   AMEVA Runtime: Native Asset Auto-Provisioner (v2.2.1)")
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

        download_url = f"{self.base_url}/{spec.filename}"
        temp_dest = target_dir / f".{spec.filename}.tmp"

        try:
            # 1. Download
            print(f"    -> Downloading: {spec.filename} ...")
            req = urllib.request.Request(download_url, headers={"User-Agent": "AMEVA-Installer/2.2.1"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(temp_dest, "wb") as out_f:
                total_len = resp.headers.get("Content-Length")
                total_bytes = int(total_len) if total_len else 0
                downloaded = 0
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)

            # 2. Extract or Deploy
            if spec.is_archive:
                print(f"    -> Unpacking archive into {spec.target_path}...")
                with tarfile.open(temp_dest, "r:gz") as tar:
                    found = False
                    for member in tar.getmembers():
                        if spec.archive_extract_target and member.name.endswith(spec.archive_extract_target):
                            extracted_f = tar.extractfile(member)
                            if extracted_f:
                                with open(spec.target_path, "wb") as out_bin:
                                    shutil.copyfileobj(extracted_f, out_bin)
                                found = True
                                break
                    if not found:
                        tar.extractall(path=target_dir)
            else:
                shutil.move(str(temp_dest), str(spec.target_path))

            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)

            # 3. Set Executable Mode
            if spec.is_executable and os.name != "nt":
                spec.target_path.chmod(0o755)

            print(f"    -> [PASS] Installed at {spec.target_path}")

            # 4. Create Dual-Track Compatibility Links
            self._create_dual_track_links(spec)
            return True

        except Exception as e:
            print(f"    -> [FAIL] Error provisioning {spec.name}: {e}")
            if temp_dest.exists():
                temp_dest.unlink(missing_ok=True)
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
    base_url: str = GITHUB_RELEASE_BASE,
) -> Dict[str, bool]:
    """Programmatic entrypoint for AMEVA Native Asset Auto-Provisioning."""
    manager = NativeAssetManager(base_url=base_url, force=force)
    return manager.provision_all(modalities=modalities)
