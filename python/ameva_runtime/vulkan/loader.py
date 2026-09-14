"""
Non-polluting Vulkan Dynamic Dispatcher & Loader (Gate 1 Compliance).
OpenSSF / CNCF Clean Engineering Standard.

Invariants:
1. Zero LD_LIBRARY_PATH pollution: Never injects /system/lib64, /vendor/lib64, or /apex paths.
2. Direct dynamic loading using RTLD_NOW | RTLD_LOCAL.
3. Entry-point validation: Obtains and verifies vkGetInstanceProcAddr.
4. Lifetime invariant: Dispatch table is bound to loader lifecycle and invalidates upon unload.
5. Fail-closed: Explicit error propagation without silent CPU fallback.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .exceptions import AmevaVulkanError


class VulkanDispatchTable:
    """
    Dynamic function pointer dispatch table resolved via vkGetInstanceProcAddr.
    Strongly bound to the parent VulkanDynamicLoader lifecycle.
    """

    def __init__(self, loader: "VulkanDynamicLoader", instance: Any = None):
        self._loader = loader
        self._instance = instance
        self._cache: Dict[str, Any] = {}

    def get_proc_addr(self, name: str) -> Optional[int]:
        """Resolves a Vulkan function pointer via vkGetInstanceProcAddr."""
        if not self._loader.is_loaded():
            raise AmevaVulkanError(
                f"[Gate 1 Safety] Cannot resolve procedure '{name}': Loader handle has been unloaded."
            )
        return self._loader.get_proc_addr(name, self._instance)

    def __getattr__(self, name: str) -> Any:
        if not self._loader.is_loaded():
            raise AmevaVulkanError(
                f"[Gate 1 Safety] Cannot invoke procedure '{name}': Loader handle has been unloaded."
            )
        if name in self._cache:
            return self._cache[name]
        addr = self.get_proc_addr(name)
        if not addr:
            raise AmevaVulkanError(f"Procedure '{name}' not found in Vulkan dispatch table.")
        self._cache[name] = addr
        return addr


class VulkanDynamicLoader:
    """
    Non-polluting dynamic Vulkan loader.
    Loads standard system libvulkan.so via dlopen(..., RTLD_NOW | RTLD_LOCAL)
    without modifying global LD_LIBRARY_PATH or polluting the process namespace.
    """

    def __init__(self, explicit_path: Optional[str] = None):
        self._explicit_path = explicit_path
        self._handle: Optional[ctypes.CDLL] = None
        self._loaded_path: Optional[str] = None
        self._vkGetInstanceProcAddr: Optional[Callable] = None

    @property
    def loaded_path(self) -> Optional[str]:
        return self._loaded_path

    def is_loaded(self) -> bool:
        return self._handle is not None and self._vkGetInstanceProcAddr is not None

    def load(self, explicit_path: Optional[str] = None) -> bool:
        """
        Dynamically loads the Vulkan library using RTLD_NOW | RTLD_LOCAL.
        Fails closed if the library cannot be loaded or if vkGetInstanceProcAddr is missing.
        """
        if self.is_loaded():
            return True

        target_path = explicit_path or self._explicit_path
        candidates = []

        if target_path:
            candidates.append(target_path)
        else:
            if sys.platform == "win32":
                candidates.extend(["vulkan-1.dll"])
            elif sys.platform == "darwin":
                candidates.extend(["libvulkan.dylib", "libvulkan.1.dylib", "libMoltenVK.dylib"])
            else:
                # Android / Linux: Bionic/Standard dynamic loader name first
                candidates.extend([
                    "libvulkan.so",
                    "libvulkan.so.1",
                    "/system/lib64/libvulkan.so",
                    "/system/lib/libvulkan.so",
                ])

        dlopen_mode = 0
        if hasattr(ctypes, "RTLD_NOW") and hasattr(ctypes, "RTLD_LOCAL"):
            dlopen_mode = ctypes.RTLD_NOW | ctypes.RTLD_LOCAL
        elif hasattr(os, "RTLD_NOW") and hasattr(os, "RTLD_LOCAL"):
            dlopen_mode = os.RTLD_NOW | os.RTLD_LOCAL
        else:
            dlopen_mode = 1

        last_error = None
        loaded_lib = None
        loaded_p = None

        for cand in candidates:
            try:
                # Direct CDLL call without mutating LD_LIBRARY_PATH
                loaded_lib = ctypes.CDLL(cand, mode=dlopen_mode)
                loaded_p = cand
                break
            except (OSError, Exception) as e:
                last_error = e
                continue

        if loaded_lib is None:
            raise AmevaVulkanError(
                f"[Gate 1 Fail-Closed] Failed to dynamically load Vulkan library from candidates {candidates}. "
                f"Underlying error: {last_error}"
            )

        # Verify vkGetInstanceProcAddr entry point
        try:
            gpa = getattr(loaded_lib, "vkGetInstanceProcAddr")
            gpa.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            gpa.restype = ctypes.c_void_p
            self._vkGetInstanceProcAddr = gpa
        except (AttributeError, ValueError) as e:
            self.unload()
            raise AmevaVulkanError(
                f"[Gate 1 Fail-Closed] vkGetInstanceProcAddr symbol missing in loaded library '{loaded_p}': {e}"
            )

        self._handle = loaded_lib
        self._loaded_path = loaded_p
        return True

    def unload(self) -> None:
        """Unloads the library handle and invalidates dispatch procedures."""
        if self._handle is not None:
            try:
                if sys.platform != "win32":
                    import _ctypes
                    if hasattr(_ctypes, "dlclose") and hasattr(self._handle, "_handle"):
                        _ctypes.dlclose(self._handle._handle)
            except Exception:
                pass
            self._handle = None
            self._loaded_path = None
            self._vkGetInstanceProcAddr = None

    def get_proc_addr(self, name: str, instance: Any = None) -> Optional[int]:
        """Resolves a Vulkan function pointer via vkGetInstanceProcAddr."""
        if not self.is_loaded() or self._vkGetInstanceProcAddr is None:
            raise AmevaVulkanError("[Gate 1 Safety] Cannot resolve procedure: Loader is not loaded.")
        inst_ptr = instance if instance else None
        c_name = name.encode("utf-8") if isinstance(name, str) else name
        addr = self._vkGetInstanceProcAddr(inst_ptr, c_name)
        return addr if addr else None

    def create_dispatch_table(self, instance: Any = None) -> VulkanDispatchTable:
        """Creates a dispatch table bound to this loader instance."""
        if not self.is_loaded():
            self.load()
        return VulkanDispatchTable(self, instance=instance)
