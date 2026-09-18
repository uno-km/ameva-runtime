# Third-Party Notices and Component Attributions

This document describes third-party open-source components associated with the `@ameva/runtime` project, detailing licensing, static linkage, and distribution in npm artifacts.

---

## 1. Node.js Node-API
- **Component**: Node-API (N-API) C Interface
- **Source**: https://github.com/nodejs/node
- **License**: MIT
- **Redistributed Source**: No
- **Statically Linked**: No (Native addon compiled against standard N-API C headers)
- **Included in npm Tarball**: No headers distributed; binary adheres to N-API ABI.

```
Copyright Joyent, Inc. and other Node contributors. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
...
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND...
```

---

## 2. LLVM Project / Termux libc++ Runtime
- **Component**: LLVM libc++ runtime (`libc++_shared.so`)
- **Source**: https://github.com/llvm/llvm-project / https://github.com/termux/termux-packages
- **License**: Apache-2.0 with LLVM Exceptions
- **Redistributed Source**: No
- **Statically Linked**: No (Addon requires dynamic loading from Termux `$PREFIX/lib/libc++_shared.so`)
- **Included in npm Tarball**: No (Host runtime environment dependency)

---

## 3. Khronos Group Vulkan Headers & Specification
- **Component**: Vulkan API Headers & ICD Interface
- **Source**: https://github.com/KhronosGroup/Vulkan-Headers
- **License**: Apache-2.0
- **Redistributed Source**: No
- **Statically Linked**: No (Dynamic loader uses `dlopen`/`dlsym` on host `libvulkan.so`)
- **Included in npm Tarball**: No

---

## 4. Subprocess Execution Compatibility Targets (Unbundled)
The execution plans (`LlamaCppExecutionPlan`, `WhisperExecutionPlan`, etc.) provide subprocess option wrappers for external user-provided executables. No third-party model weights, engine binaries, or upstream source codes are redistributed inside the `@ameva/runtime` package.
