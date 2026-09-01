# AMEVA-Vulkan-Runtime Release Notes

All notable changes and milestones for `ameva-vulkan-runtime` will be documented in this file.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and Apache-2.0 governance.

---

## [v1.0.0] - 2026-09-01
### Production Initial Release — Unified Multi-Modal Acceleration

#### 🚀 Highlights
- **Universal Multi-Modal Acceleration Core**: Unified C++20 Vulkan Hardware Abstraction Layer (HAL) accelerating STT, Vision, LLM, Diffusion, and Autograd Training.
- **Single Loader Chain Pinning**: Strict Bionic ICD `/system/lib64/libvulkan.so` dispatching, completely preventing Termux Mesa symbol collisions and SIGABRT.
- **12-Stage Probing & Auto-Recovery (V0~V11)**: Granular diagnostic hierarchy validating GPU capability from `dlopen` to SGEMM precision and E2E model pipelines with transparent CPU NEON auto-recovery.
- **Zero-Drift Hardware Quirks**: Out-of-the-box hardware bug mitigation for Qualcomm Adreno 830/750/740, ARM Mali-G78/G68/G77 (128-byte alignment), and Samsung Xclipse.
- **Cross-Platform Dual Bindings**: Full Python CFFI/ctypes and Node.js/TypeScript N-API SDK.

#### 📦 Distribution
- PyPI: `pip install ameva-vulkan-runtime`
- npm: `npm install ameva-vulkan-runtime`
- Web Docs: [https://uno-km.vercel.app/lib/vulkan/](https://uno-km.vercel.app/lib/vulkan/)
