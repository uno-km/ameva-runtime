#!/usr/bin/env python3
"""
AMEVA Runtime — Automated Release Manifest & Artifact Integrity Verifier
Calculates SHA-256 digests directly from compiled binaries/tarballs, validates AArch64 ELF
headers, and generates cryptographic release-manifest.json and .sha256 files without human/AI intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Expected bundle specifications
EXPECTED_BUNDLES: Dict[str, Dict[str, Any]] = {
    "stt": {
        "filename": "whisper-cli-vulkan-android-arm64.tar.gz",
        "binary_relpath": "bin/whisper-cli",
        "canonical_name": "whisper-cli",
        "architecture": "aarch64",
        "backend": "vulkan",
    },
    "tts": {
        "filename": "sherpa-ncnn-offline-tts-vulkan-arm64.tar.gz",
        "binary_relpath": "sherpa-ncnn-offline-tts",
        "canonical_name": "sherpa-ncnn-offline-tts",
        "architecture": "aarch64",
        "backend": "vulkan",
    },
    "diffusion": {
        "filename": "sd-cli-vulkan-android-arm64.tar.gz",
        "binary_relpath": "sd-cli-vulkan",
        "canonical_name": "sd-cli",
        "architecture": "aarch64",
        "backend": "vulkan",
    },
}


def calc_sha256(path: Path) -> str:
    """Computes SHA-256 checksum of a file in 1MB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()


def verify_elf_aarch64(stream: Any, member_name: str) -> None:
    """Verifies that the byte stream represents an ELF64 Little-Endian AArch64 binary."""
    header = stream.read(64)
    if len(header) < 20 or not header.startswith(b"\x7fELF"):
        raise ValueError(f"Member '{member_name}' is not an ELF binary")
    if header[4] != 2:
        raise ValueError(f"Member '{member_name}' is not ELF64")
    if header[5] != 1:
        raise ValueError(f"Member '{member_name}' is not Little-Endian")
    machine = int.from_bytes(header[18:20], "little")
    if machine != 183:
        raise ValueError(f"Member '{member_name}' machine ID is {machine}, expected 183 (AArch64)")


def inspect_and_verify_bundle(tar_path: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Inspects the tarball, checks internal members, and calculates internal member hashes."""
    if not tar_path.is_file():
        raise FileNotFoundError(f"Bundle file does not exist: {tar_path}")

    primary_rel = spec["binary_relpath"]
    primary_found = False
    deployed_files = {}

    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue

            extracted_stream = tar.extractfile(member)
            if extracted_stream is None:
                continue

            content = extracted_stream.read()
            m_sha = hashlib.sha256(content).hexdigest().lower()
            deployed_files[member.name] = {
                "size": member.size,
                "sha256": m_sha,
            }

            if member.name == primary_rel:
                primary_found = True
                extracted_stream.seek(0)
                verify_elf_aarch64(extracted_stream, member.name)

    if not primary_found:
        raise ValueError(
            f"Bundle '{tar_path.name}' does not contain expected primary binary: '{primary_rel}'"
        )

    return deployed_files


def update_installer_registry(installer_path: Path, bundle_shas: Dict[str, str]) -> None:
    """Safely updates NATIVE_ASSETS download_sha256 hashes in installer.py using regex."""
    if not installer_path.is_file():
        print(f"[-] Installer file not found at {installer_path}, skipping source update.")
        return

    content = installer_path.read_text(encoding="utf-8")
    modified = content

    for bundle_name, sha in bundle_shas.items():
        # Matches: "stt": AssetSpec(\n ... download_sha256="...",
        pattern = re.compile(
            rf'("{bundle_name}":\s*AssetSpec\([^)]*?download_sha256=")[0-9a-fA-F]{{64}}(")',
            re.DOTALL,
        )
        if pattern.search(modified):
            modified = pattern.sub(rf"\g<1>{sha}\g<2>", modified)
            print(f"[+] Updated installer.py NATIVE_ASSETS['{bundle_name}'] -> {sha}")
        else:
            print(f"[-] Warning: Could not match NATIVE_ASSETS entry for '{bundle_name}' in installer.py")

    if modified != content:
        installer_path.write_text(modified, encoding="utf-8")
        print(f"[+] Successfully updated {installer_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AMEVA Automated Release Manifest & Artifact Integrity Generator"
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        required=True,
        help="Directory containing compiled .tar.gz bundle assets",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="latest",
        help="Release tag name (e.g. v1.2.4)",
    )
    parser.add_argument(
        "--update-installer",
        type=Path,
        default=None,
        help="Optional path to python/ameva_runtime/installer.py to update with computed SHA-256",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="Output path for release-manifest.json",
    )

    args = parser.parse_args()
    assets_dir: Path = args.assets_dir.resolve()

    if not assets_dir.is_dir():
        print(f"[!] Error: Assets directory does not exist: {assets_dir}", file=sys.stderr)
        return 1

    print(f"[*] Analyzing release assets in: {assets_dir}")
    print(f"[*] Release Tag: {args.tag}")

    manifest_assets: Dict[str, Any] = {}
    bundle_shas: Dict[str, str] = {}
    errors: List[str] = []

    for bundle_key, spec in EXPECTED_BUNDLES.items():
        filename = spec["filename"]
        tar_path = assets_dir / filename

        if not tar_path.is_file():
            print(f"[-] Missing expected asset: {filename} (skipped or not built)")
            continue

        print(f"[*] Processing {filename} ...")
        # 1. Compute archive SHA-256
        actual_sha = calc_sha256(tar_path)
        bundle_shas[bundle_key] = actual_sha

        # 2. Write individual .sha256 file
        sha_file = assets_dir / f"{filename}.sha256"
        sha_file.write_text(f"{actual_sha}  {filename}\n", encoding="utf-8")
        print(f"    -> SHA-256: {actual_sha} (wrote {sha_file.name})")

        # 3. Verify internal bundle integrity (ELF64 AArch64)
        try:
            deployed_files = inspect_and_verify_bundle(tar_path, spec)
            print(f"    -> [PASS] Verified AArch64 ELF primary binary: {spec['binary_relpath']}")
        except Exception as e:
            err_msg = f"Integrity check failed for {filename}: {e}"
            print(f"    -> [FAIL] {err_msg}", file=sys.stderr)
            errors.append(err_msg)
            continue

        manifest_assets[filename] = {
            "bundle": bundle_key,
            "sha256": actual_sha,
            "size": tar_path.stat().st_size,
            "architecture": spec["architecture"],
            "backend": spec["backend"],
            "binary_relpath": spec["binary_relpath"],
            "canonical_name": spec["canonical_name"],
            "deployed_files": deployed_files,
        }

    if errors:
        print("\n[!] FATAL: Artifact verification failed. Release blocked:", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return 1

    if not manifest_assets:
        print("[!] Error: No valid assets found to package into manifest.", file=sys.stderr)
        return 1

    # 4. Generate unified release-manifest.json
    manifest_data = {
        "release": args.tag,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_assets": len(manifest_assets),
        "assets": manifest_assets,
    }

    out_manifest = args.output_manifest or (assets_dir / "release-manifest.json")
    out_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"\n[+] Unified release manifest generated: {out_manifest}")

    # 5. Optionally update installer.py
    if args.update_installer:
        update_installer_registry(args.update_installer.resolve(), bundle_shas)

    print("\n[+] Verification and automation complete. Zero-Human/Zero-AI SHA provenance established.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
