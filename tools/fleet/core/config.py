"""
AMEVA Vulkan Runtime - Fleet and Device Configuration
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PRESETS_DIR = BASE_DIR / 'presets'
MODELS_DIR = BASE_DIR / 'models'
BACKENDS_DIR = BASE_DIR / 'backends'
RESULTS_DIR = BASE_DIR / 'results'

SSH_KEY_PATH = os.path.expanduser('~/.ssh/id_rsa')
SSH_PORT = 8022

@dataclass
class DeviceConfig:
    id: str
    name: str
    model: str
    soc: str
    gpu_arch: str      # 'mali' or 'adreno'
    gpu_core: str      # e.g. 'Mali-G68 MP5', 'Mali-G78 MP14', 'Adreno 650'
    ip: str
    user: str
    port: int = SSH_PORT
    ssh_key: str = SSH_KEY_PATH
    remote_root: str = '/data/data/com.termux/files/home/ameva-vulkan-runtime'
    models_dir: str = '/data/data/com.termux/files/home/models'
    total_ram_gb: float = 6.0
    disk_avail_gb: float = 10.0
    optimal_threads: int = 4
    default_env: Dict[str, str] = field(default_factory=dict)

def _load_fleet_overrides() -> Dict[str, Dict[str, Any]]:
    candidates = [
        Path.cwd() / "fleet_hosts.json",
        BASE_DIR / "fleet_hosts.json",
        Path.home() / ".config" / "ameva" / "fleet_hosts.json",
    ]
    for c in candidates:
        if c.exists():
            try:
                import json
                with open(c, "r", encoding="utf-8-sig") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}

_OVERRIDES = _load_fleet_overrides()

FLEET: Dict[str, DeviceConfig] = {
    'a35': DeviceConfig(
        id='a35',
        name='Galaxy A35 5G',
        model='SM-A356N',
        soc='Exynos 1380 (s5e8835)',
        gpu_arch='mali',
        gpu_core='Mali-G68 MP5',
        ip=_OVERRIDES.get('a35', {}).get('ip', os.environ.get('AMEVA_A35_IP', '127.0.0.1')),
        user=_OVERRIDES.get('a35', {}).get('user', os.environ.get('AMEVA_A35_USER', 'termux')),
        port=_OVERRIDES.get('a35', {}).get('port', int(os.environ.get('AMEVA_A35_PORT', SSH_PORT))),
        total_ram_gb=5.3,
        disk_avail_gb=17.0,
        optimal_threads=4
    ),
    'a53': DeviceConfig(
        id='a53',
        name='Galaxy A53 5G',
        model='SM-A536N',
        soc='Exynos 1280 (s5e8825)',
        gpu_arch='mali',
        gpu_core='Mali-G68 MP4',
        ip=_OVERRIDES.get('a53', {}).get('ip', os.environ.get('AMEVA_A53_IP', '127.0.0.1')),
        user=_OVERRIDES.get('a53', {}).get('user', os.environ.get('AMEVA_A53_USER', 'termux')),
        port=_OVERRIDES.get('a53', {}).get('port', int(os.environ.get('AMEVA_A53_PORT', SSH_PORT))),
        total_ram_gb=5.3,
        disk_avail_gb=89.0,
        optimal_threads=4
    ),
    's21': DeviceConfig(
        id='s21',
        name='Galaxy S21 5G',
        model='SM-G991N',
        soc='Exynos 2100 (universal2100)',
        gpu_arch='mali',
        gpu_core='Mali-G78 MP14',
        ip=_OVERRIDES.get('s21', {}).get('ip', os.environ.get('AMEVA_S21_IP', '127.0.0.1')),
        user=_OVERRIDES.get('s21', {}).get('user', os.environ.get('AMEVA_S21_USER', 'termux')),
        port=_OVERRIDES.get('s21', {}).get('port', int(os.environ.get('AMEVA_S21_PORT', SSH_PORT))),
        total_ram_gb=7.0,
        disk_avail_gb=171.0,
        optimal_threads=4
    ),
    's20': DeviceConfig(
        id='s20',
        name='Galaxy S20+ 5G',
        model='SM-G986N',
        soc='Snapdragon 865 (SM8250)',
        gpu_arch='adreno',
        gpu_core='Adreno 650',
        ip=_OVERRIDES.get('s20', {}).get('ip', os.environ.get('AMEVA_S20_IP', '127.0.0.1')),
        user=_OVERRIDES.get('s20', {}).get('user', os.environ.get('AMEVA_S20_USER', 'termux')),
        port=_OVERRIDES.get('s20', {}).get('port', int(os.environ.get('AMEVA_S20_PORT', SSH_PORT))),
        total_ram_gb=10.0,
        disk_avail_gb=196.0,
        optimal_threads=4
    ),
    's25': DeviceConfig(
        id='s25',
        name='Galaxy S25 Flagship',
        model='SM-S931N',
        soc='Snapdragon 8 Elite (SM8750)',
        gpu_arch='adreno',
        gpu_core='Adreno 830',
        ip=_OVERRIDES.get('s25', {}).get('ip', os.environ.get('AMEVA_S25_IP', '127.0.0.1')),
        user=_OVERRIDES.get('s25', {}).get('user', os.environ.get('AMEVA_S25_USER', 'termux')),
        port=_OVERRIDES.get('s25', {}).get('port', int(os.environ.get('AMEVA_S25_PORT', SSH_PORT))),
        total_ram_gb=10.0,
        disk_avail_gb=82.0,
        optimal_threads=6
    )
}

def get_device(device_id: str) -> DeviceConfig:
    key = device_id.lower().replace('galaxy-', '').replace('+', '')
    if key not in FLEET:
        raise ValueError(f'Unknown device: {device_id}. Available: {list(FLEET.keys())}')
    return FLEET[key]
