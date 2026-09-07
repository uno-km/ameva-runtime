#!/usr/bin/env python3
"""
AMEVA Vulkan Runtime - Unified CLI Entrypoint
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path

# Add fleet dir and project root to sys.path
FLEET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FLEET_DIR.parent.parent
for p in (str(FLEET_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.config import FLEET, PRESETS_DIR, MODELS_DIR, RESULTS_DIR, get_device
from core.engine import DiffusionEngine
from core.fleet import FleetManager
try:
    from audit import audit_image
except ImportError:
    from tools.fleet.audit import audit_image

def cmd_fleet_status(args):
    print("\n==========================================================================================")
    print("                     AMEVA Vulkan Runtime - Fleet Status                                  ")
    print("==========================================================================================")
    print(f"{'ID':<6} | {'Device Name':<20} | {'AP / SoC':<24} | {'GPU Core':<16} | {'RAM':<10} | {'Disk':<8}")
    print("-" * 95)
    
    fleet_mgr = FleetManager()
    for dev_id, dev in FLEET.items():
        cmd = "echo CHIP=$(getprop ro.hardware.chipname || getprop ro.board.platform) FREE_RAM=$(free -h | grep Mem | awk '{print $7}') DISK=$(df -h /data | tail -1 | awk '{print $4}')"
        try:
            res = fleet_mgr.run_ssh(dev, cmd, timeout=5)
            out = res.stdout.strip()
            # Parse
            free_ram = "N/A"
            free_disk = "N/A"
            for token in out.split():
                if token.startswith("FREE_RAM="):
                    free_ram = token.split("=")[1]
                elif token.startswith("DISK="):
                    free_disk = token.split("=")[1]
            status_str = f"ONLINE (Free {free_ram} RAM, {free_disk} Disk)"
            print(f"{dev.id:<6} | {dev.name:<20} | {dev.soc:<24} | {dev.gpu_core:<16} | {free_ram:<10} | {free_disk:<8}")
        except Exception as e:
            print(f"{dev.id:<6} | {dev.name:<20} | {dev.soc:<24} | {dev.gpu_core:<16} | OFFLINE: {e}")
    print("=" * 95 + "\n")

def cmd_deploy(args):
    device_id = args.device
    fleet_mgr = FleetManager()
    dev = get_device(device_id)
    print(f"[*] Deploying runtime backend to {dev.name} ({dev.id})...")
    if dev.gpu_arch == "mali":
        res = fleet_mgr.deploy_mali_backend(device_id)
    elif dev.gpu_arch == "adreno":
        res = fleet_mgr.deploy_adreno_backend(device_id)
    else:
        raise ValueError(f"Unsupported GPU architecture: {dev.gpu_arch}")

    if res["success"]:
        print(f"[+] SUCCESS: {dev.gpu_arch.upper()} Vulkan backend deployed and verified on {dev.name}!")
        print(f"    Output: {res['output']}")
    else:
        print(f"[-] ERROR deploying to {dev.name}:")
        print(f"    STDOUT: {res['output']}")
        print(f"    STDERR: {res['error']}")

def cmd_run(args):
    device_id = args.device
    preset_name = args.preset
    prompt = args.prompt
    seed = args.seed
    
    preset_file = PRESETS_DIR / f"{preset_name}.json"
    if not preset_file.exists():
        print(f"[-] Preset '{preset_name}' not found. Available: {[p.stem for p in PRESETS_DIR.glob('*.json')]}")
        return
        
    with open(preset_file, "r", encoding="utf-8") as f:
        preset = json.load(f)
        
    dev = get_device(device_id)
    fleet_mgr = FleetManager()
    
    # Check registry for model URL
    registry_file = MODELS_DIR / "registry.json"
    with open(registry_file, "r", encoding="utf-8") as f:
        registry = json.load(f)
        
    model_id = preset["model_id"]
    model_info = registry.get(model_id)
    if not model_info:
        print(f"[-] Model '{model_id}' not found in registry.")
        return
        
    print(f"[*] Target: {dev.name} ({dev.gpu_core})")
    print(f"[*] Preset: {preset['display_name']} ({preset['steps']} steps, CFG {preset['cfg_scale']})")
    print(f"[*] Ensuring model '{model_info['filename']}' exists on {dev.name}...")
    
    # Ensure model on device (or check preexisting termux-diffusion cache)
    cache_check = fleet_mgr.run_ssh(dev, f"test -f ~/.cache/termux-diffusion/models/sdxs.gguf && mkdir -p {dev.models_dir} && cp -u ~/.cache/termux-diffusion/models/sdxs.gguf {dev.models_dir}/{model_info['filename']} 2>/dev/null || true")
    mdl_res = fleet_mgr.ensure_model(device_id, model_info["filename"], model_info["url"])
    print(f"    Model status: {mdl_res['status']}")
    
    # Build remote execution command
    timestamp = int(time.time())
    remote_out = f"{dev.remote_root}/output_{preset_name}_{timestamp}.png"
    local_out = RESULTS_DIR / f"{dev.id}_{preset_name}_{timestamp}.png"
    
    binary_path = f"{dev.remote_root}/bin/sd-cli-vulkan"
    shim_path = f"{dev.remote_root}/lib/libegl_shim.so" if dev.gpu_arch == "mali" else None
    lib_dir = f"{dev.remote_root}/lib" if dev.gpu_arch == "mali" else None

    # Architecture-specific hardware acceleration parameters
    backend = "clip=vulkan0,diffusion=vulkan0,vae=vulkan0" if dev.gpu_arch == "adreno" else None
    extra_env = {"GGML_VULKAN_SKIP_CHECKS": "999999999"} if dev.gpu_arch == "adreno" else None
    guidance = preset.get("guidance", 3.5 if preset_name == "sdxs" else None)
    
    engine = DiffusionEngine(binary_path=binary_path, shim_path=shim_path, lib_dir=lib_dir)
    exec_cmd = engine.build_command(
        model_path=f"{dev.models_dir}/{model_info['filename']}",
        prompt=prompt,
        output_path=remote_out,
        steps=preset["steps"],
        cfg_scale=preset["cfg_scale"],
        guidance=guidance,
        sample_method=preset["sample_method"],
        seed=seed,
        width=preset["width"],
        height=preset["height"],
        threads=dev.optimal_threads,
        vae_tiling=preset.get("vae_tiling", False),
        backend=backend,
        extra_env=extra_env
    )
    
    print(f"[*] Executing on {dev.name} GPU...")
    start_t = time.perf_counter()
    run_res = fleet_mgr.run_ssh(dev, exec_cmd, timeout=300)
    elapsed = time.perf_counter() - start_t
    
    # Parse log
    stats = engine.parse_log(run_res.stdout, run_res.stderr)
    print(f"[+] Execution completed in {elapsed:.2f}s (Engine reported: {stats['total_duration_sec']}s)")
    if stats["errors"]:
        print(f"[!] Warnings/Errors detected: {stats['errors']}")
        
    # Pull image
    print(f"[*] Downloading generated image from {dev.name}...")
    fleet_mgr.scp_from(dev, remote_out, str(local_out))
    
    if local_out.exists():
        print(f"[+] Saved image to: {local_out}")
        audit = audit_image(str(local_out))
        print(f"[+] Image Audit: Entropy={audit['entropy_bits']} bits, Low Clip={audit['clipped_low_pct']}%, High Clip={audit['clipped_high_pct']}%")
    else:
        print(f"[-] Failed to retrieve image. SSH STDOUT:\n{run_res.stdout}\nSTDERR:\n{run_res.stderr}")

def main():
    parser = argparse.ArgumentParser(description="AMEVA Vulkan Runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # status
    p_status = subparsers.add_parser("status", help="Check fleet device connectivity & resources")
    p_status.set_defaults(func=cmd_fleet_status)
    
    # deploy
    p_deploy = subparsers.add_parser("deploy", help="Deploy runtime binary to device")
    p_deploy.add_argument("--device", required=True, help="Target device id (a35, a53, s21, s20, s25)")
    p_deploy.set_defaults(func=cmd_deploy)
    
    # run
    p_run = subparsers.add_parser("run", help="Run diffusion preset on device")
    p_run.add_argument("--device", required=True, help="Target device id")
    p_run.add_argument("--preset", default="fast", help="Preset name (fast, balanced, realistic, etc.)")
    p_run.add_argument("--prompt", default="a lovely cyberpunk cat, highly detailed, 8k", help="Prompt text")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed")
    p_run.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
