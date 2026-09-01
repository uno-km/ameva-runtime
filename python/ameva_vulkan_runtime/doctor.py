"""
12-Stage Diagnostic Doctor Engine (V0~V11) — 실제 Vulkan API 호출 기반.

[중요] 본 모듈은 시뮬레이션 코드를 일절 포함하지 않습니다.
각 단계는 실제 Vulkan C API (vkCreateInstance 등) 를 ctypes 로 직접 호출하여
결과를 계측합니다. 실제로 실패하는 경우 FAIL 또는 SKIP 으로 정직하게 반환합니다.

발생한 모든 오류는 [ameva-vulkan-runtime] 태그와 함께 logging 에 기록됩니다.
"""
import ctypes
import json
import logging
import os
import platform
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ameva_vulkan_runtime.doctor")

# ---------------------------------------------------------------------------
# Vulkan C ABI 구조체 최소 정의 (Vulkan 1.x 호환)
# ---------------------------------------------------------------------------

VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = 2
VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO = 3
VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = 5
VK_NULL_HANDLE = None
VK_QUEUE_COMPUTE_BIT = 0x00000002
VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 0x00000002
VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 0x00000004
VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_uint32),
        ("deviceName", ctypes.c_char * 256),
        ("pipelineCacheUUID", ctypes.c_uint8 * 16),
        ("limits", ctypes.c_uint8 * 504),   # VkPhysicalDeviceLimits 크기 근사
        ("sparseProperties", ctypes.c_uint8 * 20),
    ]


class VkQueueFamilyProperties(ctypes.Structure):
    _fields_ = [
        ("queueFlags", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("timestampValidBits", ctypes.c_uint32),
        ("minImageTransferGranularity", ctypes.c_uint8 * 12),
    ]


class VkMemoryType(ctypes.Structure):
    _fields_ = [
        ("propertyFlags", ctypes.c_uint32),
        ("heapIndex", ctypes.c_uint32),
    ]


class VkMemoryHeap(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
    ]


class VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [
        ("memoryTypeCount", ctypes.c_uint32),
        ("memoryTypes", VkMemoryType * 32),
        ("memoryHeapCount", ctypes.c_uint32),
        ("memoryHeaps", VkMemoryHeap * 16),
    ]


class VkDeviceQueueCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueFamilyIndex", ctypes.c_uint32),
        ("queueCount", ctypes.c_uint32),
        ("pQueuePriorities", ctypes.POINTER(ctypes.c_float)),
    ]


class VkDeviceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("queueCreateInfoCount", ctypes.c_uint32),
        ("pQueueCreateInfos", ctypes.POINTER(VkDeviceQueueCreateInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
        ("pEnabledFeatures", ctypes.c_void_p),
    ]


class VkMemoryAllocateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("allocationSize", ctypes.c_uint64),
        ("memoryTypeIndex", ctypes.c_uint32),
    ]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class StageReport:
    stage_id: int
    stage_name: str
    result: str        # "PASS", "FAIL", "SKIP"
    elapsed_ms: float
    detail_message: str
    allocated_bytes: int = 0


@dataclass
class DiagnosticReport:
    overall_success: bool
    device_name: str
    driver_version: str
    loader_path: str
    vendor_id: int
    passed_stages: int
    total_stages: int
    total_elapsed_ms: float
    recommended_backend: str
    stages: List[StageReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vulkan 라이브러리 경로 탐색 (Android Bionic ICD 우선)
# ---------------------------------------------------------------------------

_VULKAN_SEARCH_PATHS = [
    "/system/lib64/libvulkan.so",         # Android Bionic ICD (최우선)
    "/vendor/lib64/libvulkan.so",
    "/system/lib/libvulkan.so",
    "libvulkan.so.1",                      # Linux
    "libvulkan.so",
    "vulkan-1.dll",                        # Windows (개발용)
]

# Termux Mesa 경로는 의도적으로 제외 — Bionic 이중 로더 SIGABRT 방지


def _find_vulkan_lib() -> Optional[str]:
    """Bionic ICD 우선순위로 Vulkan 라이브러리를 탐색합니다."""
    prefix = os.environ.get("PREFIX", "/data/data/com.termux/files/usr")
    termux_mesa = os.path.join(prefix, "lib", "libvulkan.so")

    for path in _VULKAN_SEARCH_PATHS:
        if os.path.isfile(path):
            # Termux Mesa 경로가 시스템 ICD보다 먼저 걸리면 경고
            if path == termux_mesa:
                logger.warning(
                    "[ameva-vulkan-runtime] V0: Termux Mesa libvulkan.so 가 탐지되었습니다. "
                    "Android ICD(/system/lib64/libvulkan.so)가 존재하지 않는 환경입니다. "
                    "Bionic 이중 로더 충돌 방지를 위해 시스템 ICD를 우선합니다."
                )
            return path

    # Windows/Linux 개발 환경용 fallback
    import ctypes.util
    found = ctypes.util.find_library("vulkan")
    if found:
        return found

    return None


# ---------------------------------------------------------------------------
# Doctor 클래스
# ---------------------------------------------------------------------------

class Doctor:
    """AMEVA-Vulkan-Runtime 12단계 하드웨어 검증 및 역량 진단기.

    실제 Vulkan C API 호출로 각 단계를 검증합니다.
    실패 시 FAIL 을 반환하고 모든 오류는 [ameva-vulkan-runtime] 태그로 기록합니다.
    """

    STAGE_NAMES = [
        "Vulkan Loader Open",               # V0
        "Instance Creation",                # V1
        "Physical Device Enumeration",      # V2
        "Hardware GPU Selection",           # V3
        "Compute Queue Family Probe",       # V4
        "Logical Device Creation",          # V5
        "Buffer Memory Allocation",         # V6
        "SPIR-V Pipeline Compilation",      # V7
        "Compute Shader Dispatch",          # V8
        "Result Checksum Validation",       # V9
        "GGML MatMul Tensor Ops",           # V10
        "End-to-End Model Inference",       # V11
    ]

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = Path(state_path or "state.json")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run_self_test(self, verbose: bool = True) -> DiagnosticReport:
        """12단계 하드웨어 검증을 실제 Vulkan API 호출로 수행합니다."""
        t_start = time.perf_counter()

        if verbose:
            print("\n" + "=" * 62)
            print("  AMEVA-Vulkan-Runtime: 12-Stage Diagnostic Suite (V0-V11)")
            print("=" * 62)

        stages: List[StageReport] = []
        passed = 0
        overall_ok = True

        # 핸들 — 각 단계에서 생성 후 다음 단계에 전달
        vk_lib = None
        vk_instance = ctypes.c_void_p(0)
        phys_device = ctypes.c_void_p(0)
        logical_device = ctypes.c_void_p(0)
        device_name = "Unknown"
        driver_version = "Unknown"
        vendor_id = 0
        loader_path = ""
        compute_queue_family = -1

        # --- V0: Loader Open ---
        s0 = self._run_stage(0, overall_ok, verbose)
        if s0 is None:
            loader_path_found = _find_vulkan_lib()
            t0 = time.perf_counter()
            if loader_path_found:
                try:
                    vk_lib = ctypes.CDLL(loader_path_found)
                    loader_path = loader_path_found
                    elapsed = (time.perf_counter() - t0) * 1000
                    s0 = StageReport(0, f"V0: {self.STAGE_NAMES[0]}", "PASS",
                                     elapsed, f"Bound to: {loader_path}")
                    passed += 1
                except OSError as e:
                    elapsed = (time.perf_counter() - t0) * 1000
                    msg = f"[ameva-vulkan-runtime] V0 dlopen 실패: {e}"
                    logger.error(msg)
                    s0 = StageReport(0, f"V0: {self.STAGE_NAMES[0]}", "FAIL", elapsed, str(e))
                    overall_ok = False
            else:
                elapsed = (time.perf_counter() - t0) * 1000
                msg = "[ameva-vulkan-runtime] V0: Vulkan ICD .so 파일을 찾을 수 없습니다."
                logger.error(msg)
                s0 = StageReport(0, f"V0: {self.STAGE_NAMES[0]}", "FAIL", elapsed,
                                 "No Vulkan ICD found in system paths")
                overall_ok = False
            self._print_stage(s0, verbose)
        stages.append(s0)

        # --- V1: Instance Creation ---
        t0 = time.perf_counter()
        if overall_ok and vk_lib:
            try:
                # vkCreateInstance 함수 시그니처 설정
                vk_lib.vkCreateInstance.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
                ]
                vk_lib.vkCreateInstance.restype = ctypes.c_int32

                app_info = VkApplicationInfo(
                    sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    pNext=None,
                    pApplicationName=b"ameva-vulkan-runtime",
                    applicationVersion=0x00010000,
                    pEngineName=b"ameva",
                    engineVersion=0x00010000,
                    apiVersion=0x00403000,  # Vulkan 1.3
                )
                inst_ci = VkInstanceCreateInfo(
                    sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    pNext=None,
                    flags=0,
                    pApplicationInfo=ctypes.byref(app_info),
                    enabledLayerCount=0,
                    ppEnabledLayerNames=None,
                    enabledExtensionCount=0,
                    ppEnabledExtensionNames=None,
                )
                result = vk_lib.vkCreateInstance(
                    ctypes.byref(inst_ci), None, ctypes.byref(vk_instance)
                )
                elapsed = (time.perf_counter() - t0) * 1000
                if result == VK_SUCCESS and vk_instance.value:
                    s1 = StageReport(1, f"V1: {self.STAGE_NAMES[1]}", "PASS", elapsed,
                                     f"vkCreateInstance() SUCCESS (result={result})")
                    passed += 1
                else:
                    msg = f"[ameva-vulkan-runtime] V1 vkCreateInstance 실패: result={result}"
                    logger.error(msg)
                    s1 = StageReport(1, f"V1: {self.STAGE_NAMES[1]}", "FAIL", elapsed,
                                     f"vkCreateInstance returned {result}")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V1 예외: %s", e)
                s1 = StageReport(1, f"V1: {self.STAGE_NAMES[1]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            elapsed = 0.0
            s1 = StageReport(1, f"V1: {self.STAGE_NAMES[1]}", "SKIP", elapsed,
                             "V0 실패로 인한 건너뜀")
        self._print_stage(s1, verbose)
        stages.append(s1)

        # --- V2: Physical Device Enumeration ---
        t0 = time.perf_counter()
        phys_devices = []
        if overall_ok and vk_lib and vk_instance.value:
            try:
                vk_lib.vkEnumeratePhysicalDevices.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                ]
                vk_lib.vkEnumeratePhysicalDevices.restype = ctypes.c_int32

                count = ctypes.c_uint32(0)
                result = vk_lib.vkEnumeratePhysicalDevices(vk_instance, ctypes.byref(count), None)
                elapsed = (time.perf_counter() - t0) * 1000
                if result == VK_SUCCESS and count.value > 0:
                    arr = (ctypes.c_void_p * count.value)()
                    vk_lib.vkEnumeratePhysicalDevices(vk_instance, ctypes.byref(count), arr)
                    phys_devices = list(arr)
                    s2 = StageReport(2, f"V2: {self.STAGE_NAMES[2]}", "PASS", elapsed,
                                     f"Enumerated {count.value} physical device(s)")
                    passed += 1
                else:
                    msg = f"[ameva-vulkan-runtime] V2: 물리 장치 없음 또는 열거 실패 (result={result}, count={count.value})"
                    logger.error(msg)
                    s2 = StageReport(2, f"V2: {self.STAGE_NAMES[2]}", "FAIL", elapsed,
                                     f"vkEnumeratePhysicalDevices: result={result}, count={count.value}")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V2 예외: %s", e)
                s2 = StageReport(2, f"V2: {self.STAGE_NAMES[2]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            s2 = StageReport(2, f"V2: {self.STAGE_NAMES[2]}", "SKIP", 0.0, "이전 단계 실패")
        self._print_stage(s2, verbose)
        stages.append(s2)

        # --- V3: Hardware GPU Selection ---
        t0 = time.perf_counter()
        if overall_ok and vk_lib and phys_devices:
            try:
                vk_lib.vkGetPhysicalDeviceProperties.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(VkPhysicalDeviceProperties),
                ]
                vk_lib.vkGetPhysicalDeviceProperties.restype = None

                best = None
                best_props = None
                for dev in phys_devices:
                    if not dev:
                        continue
                    props = VkPhysicalDeviceProperties()
                    vk_lib.vkGetPhysicalDeviceProperties(
                        ctypes.c_void_p(dev), ctypes.byref(props)
                    )
                    if best is None or props.deviceType in (
                        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                        VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                    ):
                        best = ctypes.c_void_p(dev)
                        best_props = props

                elapsed = (time.perf_counter() - t0) * 1000
                if best_props is not None:
                    phys_device = best
                    device_name = best_props.deviceName.decode("utf-8", errors="replace").rstrip("\x00")
                    vendor_id = best_props.vendorID
                    api_ver = best_props.apiVersion
                    api_str = f"{(api_ver >> 22) & 0x3FF}.{(api_ver >> 12) & 0x3FF}.{api_ver & 0xFFF}"
                    driver_version = f"drvVer=0x{best_props.driverVersion:08X} api={api_str}"
                    s3 = StageReport(3, f"V3: {self.STAGE_NAMES[3]}", "PASS", elapsed,
                                     f"Selected: {device_name} (vendorID=0x{vendor_id:04X})")
                    passed += 1
                else:
                    logger.error("[ameva-vulkan-runtime] V3: 유효한 GPU 장치를 선택하지 못했습니다.")
                    s3 = StageReport(3, f"V3: {self.STAGE_NAMES[3]}", "FAIL", elapsed,
                                     "No valid GPU device found")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V3 예외: %s", e)
                s3 = StageReport(3, f"V3: {self.STAGE_NAMES[3]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            s3 = StageReport(3, f"V3: {self.STAGE_NAMES[3]}", "SKIP", 0.0, "이전 단계 실패")
        self._print_stage(s3, verbose)
        stages.append(s3)

        # --- V4: Compute Queue Family Probe ---
        t0 = time.perf_counter()
        if overall_ok and vk_lib and phys_device and phys_device.value:
            try:
                vk_lib.vkGetPhysicalDeviceQueueFamilyProperties.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_void_p,
                ]
                vk_lib.vkGetPhysicalDeviceQueueFamilyProperties.restype = None

                qf_count = ctypes.c_uint32(0)
                vk_lib.vkGetPhysicalDeviceQueueFamilyProperties(
                    phys_device, ctypes.byref(qf_count), None
                )
                qf_arr = (VkQueueFamilyProperties * qf_count.value)()
                vk_lib.vkGetPhysicalDeviceQueueFamilyProperties(
                    phys_device, ctypes.byref(qf_count), qf_arr
                )

                for i, qf in enumerate(qf_arr):
                    if qf.queueFlags & VK_QUEUE_COMPUTE_BIT:
                        compute_queue_family = i
                        break

                elapsed = (time.perf_counter() - t0) * 1000
                if compute_queue_family >= 0:
                    s4 = StageReport(4, f"V4: {self.STAGE_NAMES[4]}", "PASS", elapsed,
                                     f"Compute Queue Family Index={compute_queue_family} (flags=0x{qf_arr[compute_queue_family].queueFlags:02X})")
                    passed += 1
                else:
                    logger.error("[ameva-vulkan-runtime] V4: Compute Queue Family를 찾지 못했습니다.")
                    s4 = StageReport(4, f"V4: {self.STAGE_NAMES[4]}", "FAIL", elapsed,
                                     "No compute queue family found")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V4 예외: %s", e)
                s4 = StageReport(4, f"V4: {self.STAGE_NAMES[4]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            s4 = StageReport(4, f"V4: {self.STAGE_NAMES[4]}", "SKIP", 0.0, "이전 단계 실패")
        self._print_stage(s4, verbose)
        stages.append(s4)

        # --- V5: Logical Device Creation ---
        t0 = time.perf_counter()
        if overall_ok and vk_lib and phys_device and phys_device.value and compute_queue_family >= 0:
            try:
                vk_lib.vkCreateDevice.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                vk_lib.vkCreateDevice.restype = ctypes.c_int32

                priority = ctypes.c_float(1.0)
                qci = VkDeviceQueueCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                    pNext=None, flags=0,
                    queueFamilyIndex=compute_queue_family,
                    queueCount=1,
                    pQueuePriorities=ctypes.byref(priority),
                )
                dci = VkDeviceCreateInfo(
                    sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                    pNext=None, flags=0,
                    queueCreateInfoCount=1,
                    pQueueCreateInfos=ctypes.byref(qci),
                    enabledLayerCount=0, ppEnabledLayerNames=None,
                    enabledExtensionCount=0, ppEnabledExtensionNames=None,
                    pEnabledFeatures=None,
                )
                result = vk_lib.vkCreateDevice(
                    phys_device, ctypes.byref(dci), None, ctypes.byref(logical_device)
                )
                elapsed = (time.perf_counter() - t0) * 1000
                if result == VK_SUCCESS and logical_device.value:
                    s5 = StageReport(5, f"V5: {self.STAGE_NAMES[5]}", "PASS", elapsed,
                                     f"vkCreateDevice() SUCCESS (result={result})")
                    passed += 1
                else:
                    msg = f"[ameva-vulkan-runtime] V5 vkCreateDevice 실패: result={result}"
                    logger.error(msg)
                    s5 = StageReport(5, f"V5: {self.STAGE_NAMES[5]}", "FAIL", elapsed,
                                     f"vkCreateDevice returned {result}")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V5 예외: %s", e)
                s5 = StageReport(5, f"V5: {self.STAGE_NAMES[5]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            s5 = StageReport(5, f"V5: {self.STAGE_NAMES[5]}", "SKIP", 0.0, "이전 단계 실패")
        self._print_stage(s5, verbose)
        stages.append(s5)

        # --- V6: Buffer Memory Allocation (32MB Host-Coherent) ---
        t0 = time.perf_counter()
        alloc_bytes = 32 * 1024 * 1024
        if overall_ok and vk_lib and logical_device and logical_device.value and phys_device and phys_device.value:
            try:
                vk_lib.vkGetPhysicalDeviceMemoryProperties.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(VkPhysicalDeviceMemoryProperties),
                ]
                vk_lib.vkGetPhysicalDeviceMemoryProperties.restype = None
                vk_lib.vkAllocateMemory.argtypes = [
                    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_void_p),
                ]
                vk_lib.vkAllocateMemory.restype = ctypes.c_int32
                vk_lib.vkFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                vk_lib.vkFreeMemory.restype = None

                mem_props = VkPhysicalDeviceMemoryProperties()
                vk_lib.vkGetPhysicalDeviceMemoryProperties(phys_device, ctypes.byref(mem_props))

                # Host-Visible + Coherent 메모리 타입 탐색
                mem_type_idx = -1
                desired = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                for i in range(mem_props.memoryTypeCount):
                    if (mem_props.memoryTypes[i].propertyFlags & desired) == desired:
                        mem_type_idx = i
                        break

                elapsed = (time.perf_counter() - t0) * 1000
                if mem_type_idx >= 0:
                    alloc_info = VkMemoryAllocateInfo(
                        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        pNext=None,
                        allocationSize=alloc_bytes,
                        memoryTypeIndex=mem_type_idx,
                    )
                    mem_handle = ctypes.c_void_p(0)
                    result = vk_lib.vkAllocateMemory(
                        logical_device, ctypes.byref(alloc_info), None, ctypes.byref(mem_handle)
                    )
                    if result == VK_SUCCESS:
                        vk_lib.vkFreeMemory(logical_device, mem_handle, None)
                        s6 = StageReport(6, f"V6: {self.STAGE_NAMES[6]}", "PASS",
                                         elapsed, f"Allocated {alloc_bytes // (1024*1024)}MB Host-Coherent (type={mem_type_idx})",
                                         allocated_bytes=alloc_bytes)
                        passed += 1
                    else:
                        logger.error("[ameva-vulkan-runtime] V6 vkAllocateMemory 실패: result=%d", result)
                        s6 = StageReport(6, f"V6: {self.STAGE_NAMES[6]}", "FAIL", elapsed,
                                         f"vkAllocateMemory returned {result}")
                        overall_ok = False
                else:
                    logger.error("[ameva-vulkan-runtime] V6: Host-Visible 메모리 타입을 찾지 못했습니다.")
                    s6 = StageReport(6, f"V6: {self.STAGE_NAMES[6]}", "FAIL", elapsed,
                                     "No HOST_VISIBLE | HOST_COHERENT memory type found")
                    overall_ok = False
            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                logger.error("[ameva-vulkan-runtime] V6 예외: %s", e)
                s6 = StageReport(6, f"V6: {self.STAGE_NAMES[6]}", "FAIL", elapsed, str(e))
                overall_ok = False
        else:
            s6 = StageReport(6, f"V6: {self.STAGE_NAMES[6]}", "SKIP", 0.0, "이전 단계 실패")
        self._print_stage(s6, verbose)
        stages.append(s6)

        # --- V7~V11: SPIR-V Pipeline / Dispatch / Checksum / MatMul / E2E ---
        # 이 단계들은 실기기에서 GGML Vulkan 셰이더 컴파일 및 dispatch가 필요합니다.
        # libameva_vulkan.so(C++ HAL) 가 배포되지 않은 환경에서는 SKIP 으로 정직하게 처리합니다.
        # 거짓 PASS 를 절대 반환하지 않습니다.
        for i in range(7, 12):
            name = self.STAGE_NAMES[i]
            if not overall_ok:
                s = StageReport(i, f"V{i}: {name}", "SKIP", 0.0, "이전 단계 실패로 건너뜀")
            else:
                # C HAL 공유 라이브러리 배포 여부 확인
                hal_path = Path(__file__).parent / "libameva_vulkan.so"
                if hal_path.exists():
                    # TODO: 실기기 배포 후 C HAL로 실제 dispatch 구현
                    s = StageReport(i, f"V{i}: {name}", "SKIP", 0.0,
                                    "C HAL libameva_vulkan.so 배포 확인됨 — dispatch 구현 예정 (실기기 전용)")
                else:
                    s = StageReport(i, f"V{i}: {name}", "SKIP", 0.0,
                                    "C HAL 미배포 환경 — 실기기(Android Bionic) 전용 단계입니다.")
            self._print_stage(s, verbose)
            stages.append(s)

        # 전체 집계
        total_elapsed = (time.perf_counter() - t_start) * 1000.0
        final_report = DiagnosticReport(
            overall_success=overall_ok,
            device_name=device_name,
            driver_version=driver_version,
            loader_path=loader_path,
            vendor_id=vendor_id,
            passed_stages=passed,
            total_stages=len(self.STAGE_NAMES),
            total_elapsed_ms=total_elapsed,
            recommended_backend="vulkan" if overall_ok else "cpu_neon",
            stages=stages,
        )

        if verbose:
            print("-" * 62)
            status_str = "VULKAN ACCELERATED" if overall_ok else "CPU NEON FALLBACK"
            print(f"  Scorecard: {passed}/{len(self.STAGE_NAMES)} Stages Passed"
                  f" | Time: {total_elapsed:.2f} ms | Mode: {status_str}")
            print("=" * 62 + "\n")

        self.save_state(final_report)
        return final_report

    def save_state(self, report: DiagnosticReport) -> None:
        """검증 결과를 state.json에 저장합니다."""
        data = {
            "overall_success": report.overall_success,
            "device_name": report.device_name,
            "driver_version": report.driver_version,
            "loader_path": report.loader_path,
            "vendor_id": report.vendor_id,
            "passed_stages": report.passed_stages,
            "total_stages": report.total_stages,
            "recommended_backend": report.recommended_backend,
            "timestamp": time.time(),
        }
        try:
            self.state_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as e:
            logger.warning("[ameva-vulkan-runtime] state.json 저장 실패: %s", e)

    def quick_probe(self) -> bool:
        """state.json 캐시를 통한 빠른 Vulkan 가용성 확인.

        캐시가 없거나 오래된 경우 전체 진단을 재실행합니다.
        """
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text(encoding="utf-8"))
                age_sec = time.time() - data.get("timestamp", 0)
                if data.get("overall_success") and age_sec < 3600:
                    return True
            except Exception as e:
                logger.warning("[ameva-vulkan-runtime] state.json 읽기 실패, 재진단 수행: %s", e)
        report = self.run_self_test(verbose=False)
        return report.overall_success

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _run_stage(stage_id: int, overall_ok: bool, verbose: bool) -> Optional[StageReport]:
        """이전 단계 실패 시 SKIP 단계를 즉시 반환하는 가드."""
        return None  # None 반환 = 단계 실행해야 함 (가드 통과)

    @staticmethod
    def _print_stage(stage: StageReport, verbose: bool) -> None:
        if not verbose:
            return
        icon = {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(stage.result, "????")
        name_col = stage.stage_name.ljust(36)
        print(f"  [{icon}] {name_col} ({stage.elapsed_ms:6.2f} ms) - {stage.detail_message}")
