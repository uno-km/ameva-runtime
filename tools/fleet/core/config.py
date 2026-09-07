"""
AMEVA Vulkan Runtime - Fleet and Device Configuration
"""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
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

FLEET: Dict[str, DeviceConfig] = {
    'a35': DeviceConfig(
        id='a35',
        name='Galaxy A35 5G',
        model='SM-A356N',
        soc='Exynos 1380 (s5e8835)',
        gpu_arch='mali',
        gpu_core='Mali-G68 MP5',
        ip='100.106.251.21',
        user='u0_a30',
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
        ip='100.77.47.37',
        user='u0_a306',
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
        ip='100.83.82.60',
        user='u0_a328',
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
        ip='100.106.99.81',
        user='u0_a37',
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
        ip='100.106.0.38',
        user='u0_a466',
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
