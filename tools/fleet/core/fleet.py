"""
AMEVA Vulkan Runtime - Fleet Management and Remote Execution
"""
import os
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from core.config import FLEET, get_device, DeviceConfig, PRESETS_DIR, MODELS_DIR, RESULTS_DIR
from core.engine import DiffusionEngine, RunResult

class FleetManager:
    def __init__(self):
        self.results_dir = RESULTS_DIR
        self.results_dir.mkdir(exist_ok=True, parents=True)

    def run_ssh(self, device: DeviceConfig, command: str, timeout: int = 120) -> subprocess.CompletedProcess:
        ssh_cmd = [
            "ssh", "-p", str(device.port),
            "-i", device.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=5",
            "-o", "StrictHostKeyChecking=no",
            f"{device.user}@{device.ip}",
            command
        ]
        return subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=timeout)

    def scp_to(self, device: DeviceConfig, local_path: str, remote_path: str, timeout: int = 180) -> subprocess.CompletedProcess:
        scp_cmd = [
            "scp", "-P", str(device.port),
            "-i", device.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            local_path,
            f"{device.user}@{device.ip}:{remote_path}"
        ]
        return subprocess.run(scp_cmd, capture_output=True, text=True, timeout=timeout)

    def scp_from(self, device: DeviceConfig, remote_path: str, local_path: str, timeout: int = 60) -> subprocess.CompletedProcess:
        scp_cmd = [
            "scp", "-P", str(device.port),
            "-i", device.ssh_key,
            "-o", "BatchMode=yes",
            "-o", "StrictHostKeyChecking=no",
            f"{device.user}@{device.ip}:{remote_path}",
            local_path
        ]
        return subprocess.run(scp_cmd, capture_output=True, text=True, timeout=timeout)

    def deploy_mali_backend(self, device_id: str) -> Dict[str, Any]:
        """Deploy Mali Vulkan binary + Shim V3 to target device."""
        device = get_device(device_id)
        if device.gpu_arch != "mali":
            raise ValueError(f"Device {device_id} is not Mali architecture (it is {device.gpu_arch})")

        print(f"[*] Deploying Mali backend to {device.name} ({device.ip})...")
        
        # 1. Create remote directories
        self.run_ssh(device, f"mkdir -p {device.remote_root}/bin {device.remote_root}/lib {device.models_dir}")

        # 2. Check if local gz and libs exist
        local_gz = Path(__file__).parent.parent / "backends" / "mali" / "bin" / "sd-cli-vulkan.gz"
        local_shim = Path(__file__).parent.parent / "backends" / "mali" / "lib" / "libegl_shim.so"
        local_omp = Path(__file__).parent.parent / "backends" / "mali" / "lib" / "libomp.so"

        # 3. SCP gzipped binary, shim, and libomp for complete zero-dependency execution
        print(f"[*] Uploading compressed binary (38MB), shim, and libomp to {device.name}...")
        self.scp_to(device, str(local_gz), f"{device.remote_root}/bin/sd-cli-vulkan.gz")
        self.scp_to(device, str(local_shim), f"{device.remote_root}/lib/libegl_shim.so")
        if local_omp.exists():
            self.scp_to(device, str(local_omp), f"{device.remote_root}/lib/libomp.so")

        # 4. Decompress on device, make executable, and verify
        print(f"[*] Decompressing and verifying on {device.name}...")
        verify_cmd = f"""
        cd {device.remote_root}/bin && gzip -d -f sd-cli-vulkan.gz && chmod +x sd-cli-vulkan
        LD_LIBRARY_PATH={device.remote_root}/lib:$LD_LIBRARY_PATH LD_PRELOAD={device.remote_root}/lib/libegl_shim.so {device.remote_root}/bin/sd-cli-vulkan --help | head -n 4
        """
        res = self.run_ssh(device, verify_cmd, timeout=30)
        success = res.returncode == 0 and "stable-diffusion.cpp" in res.stdout
        
        return {
            "device": device.id,
            "success": success,
            "output": res.stdout.strip(),
            "error": res.stderr.strip()
        }

    def deploy_adreno_backend(self, device_id: str) -> Dict[str, Any]:
        """Deploy Adreno Vulkan binary to target Snapdragon/Adreno device."""
        device = get_device(device_id)
        if device.gpu_arch != "adreno":
            raise ValueError(f"Device {device_id} is not Adreno architecture (it is {device.gpu_arch})")

        print(f"[*] Deploying Adreno backend to {device.name} ({device.ip})...")

        # 1. Create remote directories
        self.run_ssh(device, f"mkdir -p {device.remote_root}/bin {device.models_dir}")

        # 2. Check local gzipped Adreno binary
        local_gz = Path(__file__).parent.parent / "backends" / "adreno" / "bin" / "sd-cli-vulkan.gz"
        if not local_gz.exists():
            raise FileNotFoundError(f"Adreno binary package not found at: {local_gz}")

        # 3. SCP gzipped binary to device
        print(f"[*] Uploading compressed Adreno binary ({local_gz.stat().st_size / (1024*1024):.1f}MB) to {device.name}...")
        self.scp_to(device, str(local_gz), f"{device.remote_root}/bin/sd-cli-vulkan.gz")

        # 4. Decompress, make executable, and verify via healthcheck
        print(f"[*] Decompressing and verifying on {device.name}...")
        verify_cmd = f"""
        cd {device.remote_root}/bin && gzip -d -f sd-cli-vulkan.gz && chmod +x sd-cli-vulkan
        GGML_VULKAN_SKIP_CHECKS=999999999 {device.remote_root}/bin/sd-cli-vulkan --help | head -n 4
        """
        res = self.run_ssh(device, verify_cmd, timeout=30)
        success = res.returncode == 0 and "stable-diffusion.cpp" in res.stdout

        return {
            "device": device.id,
            "success": success,
            "output": res.stdout.strip(),
            "error": res.stderr.strip()
        }

    def ensure_model(self, device_id: str, model_filename: str, url: str) -> Dict[str, Any]:
        """Check if model exists on device, download via curl if missing."""
        device = get_device(device_id)
        remote_model_path = f"{device.models_dir}/{model_filename}"

        check_cmd = f"test -f {remote_model_path} && ls -lh {remote_model_path}"
        res = self.run_ssh(device, check_cmd)
        if res.returncode == 0:
            return {"status": "cached", "path": remote_model_path, "details": res.stdout.strip()}

        print(f"[*] Model {model_filename} not found on {device.name}. Downloading from HuggingFace...")
        dl_cmd = f"mkdir -p {device.models_dir} && curl -L -C - -o {remote_model_path} {url}"
        dl_res = self.run_ssh(device, dl_cmd, timeout=600)
        return {
            "status": "downloaded" if dl_res.returncode == 0 else "failed",
            "path": remote_model_path,
            "output": dl_res.stdout.strip(),
            "error": dl_res.stderr.strip()
        }
