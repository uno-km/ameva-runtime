# AMEVA-Runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/@ameva/runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime (with BitNet 1.58-bit Vulkan Compute) for Mobile & Edge

## Installation

```bash
npm install @ameva/runtime
```

## Quickstart & Recommended Invocation Workflow

```typescript
import { Doctor, createContext, isAvailable } from '@ameva/runtime';

// 1. First run or periodic validation: Execute real GPU diagnostic & issue certificate
const doctor = new Doctor();
const report = await doctor.runSelfTest(false);
console.log(`GPU: ${report.deviceName} | Compute Certified: ${report.computeCertified}`);

// 2. Synchronous context creation: Consumes the validated 24h native certificate
const ctx = createContext({ device: "auto" });
console.log(`Selected Backend: ${ctx.selectedBackend}`); // "vulkan" on certified hardware
```

## Architecture & Certification Contracts

AMEVA-Runtime enforces a strict separation between active hardware probing and synchronous state consumption:

### 1. Active Probing (`Doctor.runSelfTest()`)
- Directly dispatches into `ameva_native.node` via N-API and executes official Vulkan C ABI diagnostics (V0 through V9).
- Issues `vulkan_state.json` containing hardware fingerprint metadata (`vendorId`, `deviceId`, `apiVersion`, `driverVersion`, `loaderPath`), recorded SHA-256 identifiers for relevant artifacts, and ISO-8601 UTC timestamp (`verifiedAt`).
- Note: The state file is not cryptographically signed.

### 2. State Consumption (`isAvailable()`, `createContext()`)
- `createContext()` does **not** execute heavy GPU diagnostics on each synchronous instantiation.
- Instead, `createContext({ device: "auto" })` and `isAvailable()` evaluate the cached `vulkan_state.json`:
  - Enforces 24-hour TTL (`verifiedAt`).
  - Enforces origin verification (`verificationSource === "native_c_hal"`).
  - Enforces `computeCertified === true` and `diagnosticScope === "compute"`.
  - Performs **Certificate Metadata Validation** against the local driver loader path and canonical schema.
- Note: `quickProbe()` trusts the verified **Native Compute certification state** issued by Native HAL; it does **not** perform real-time GPU hardware re-querying during synchronous context instantiation.

### 3. Certification Status Separation
- **Doctor Compute Certified**: Strictly evaluates to `true` when stages V0 (Loader Open) through V9 (Result Checksum Validation) all individually PASS.
- **Doctor Model Certified**: Strictly evaluates to `false` in current releases, as high-level model runtime graphs (V10–V11) are deferred to dedicated engines.
- **Modality Engine Benchmarks**: Real-device performance figures (LLaMA-3.2, SDXS, Whisper) are empirical measurements from separate model runtime executions, decoupled from Doctor V11 PASS status.
- **CPU Reference GEMM**: `matmulF32` is a native host C reference implementation (`backend: "cpu_reference"`, `vulkanDispatch: "NOT_PERFORMED"`).

## Controlled Subprocess Execution Engine (Phase N2)

AMEVA-Runtime provides a hardened, single-shot subprocess execution layer (`executeSubprocess`) designed for running on-device AI binaries (e.g. LLaMA.cpp, Whisper.cpp):
- **Deterministic Invocation**: Strictly enforces `shell: false`, array arguments, and pre-validated absolute executable paths.
- **Resource Limits & OOM Protection**: Independent stream bounds (`maxStdoutBytes`, `maxStderrBytes`, `maxTelemetryBytes`). Exceeding boundaries triggers `OUTPUT_LIMIT_EXCEEDED` with graceful `SIGTERM` followed by `SIGKILL`.
- **First-Cause-Wins Termination**: Clear, unambiguous primary termination tracking across timeouts, abort signals, and limit violations.
- **Structured Telemetry (FD 3)**: Acceleration backends (`backendConfirmed`) are verified exclusively via schema-validated JSON Lines on dedicated file descriptor 3 (`AMEVA_TELEMETRY_FD=3`). Unstructured stdout/stderr strings are kept as unconfirmed diagnostic hints only.

```typescript
import { executeSubprocess, LlamaCppExecutionPlan, createContext } from '@ameva/runtime';

const ctx = createContext({ device: "auto" });
const plan = LlamaCppExecutionPlan.create("llama-cli", ctx);

const result = await executeSubprocess(plan.toSubprocessOptions("/data/data/com.termux/files/usr/bin/llama-cli", [
  "-m", "/sdcard/models/llama-3.2-1b.gguf",
  "-p", "Hello world",
  "-n", "32"
], {
  timeoutMs: 15000,
  maxStdoutBytes: 2 * 1024 * 1024
}));

console.log(`Exit Code: ${result.exitCode}, Backend Confirmed: ${result.backendConfirmed}`);
```

## Documentation
- [Official Documentation](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
