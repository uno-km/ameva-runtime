/**
 * AMEVA Vulkan Acceleration Doctor for Node.js (Honest Hardware Probing)
 */
const fs = require('fs');
const path = require('path');
const os = require('os');

const VULKAN_SEARCH_PATHS = [
  '/system/lib64/libvulkan.so',
  '/vendor/lib64/libvulkan.so',
  '/system/lib/libvulkan.so',
  'libvulkan.so.1',
  'libvulkan.so',
  'vulkan-1.dll'
];

const MAX_CERT_AGE_MS = 24 * 60 * 60 * 1000; // 24 Hours TTL for native hardware certification

function getDefaultCachePath() {
  const cacheDir = path.join(os.homedir(), '.cache', 'ameva');
  try {
    fs.mkdirSync(cacheDir, { recursive: true });
    return path.join(cacheDir, 'vulkan_state.json');
  } catch (e) {
    return path.join(os.tmpdir(), 'ameva_vulkan_state.json');
  }
}

function findVulkanLib() {
  if (process.env.AMEVA_VULKAN_LIB && fs.existsSync(process.env.AMEVA_VULKAN_LIB)) {
    return process.env.AMEVA_VULKAN_LIB;
  }
  for (const p of VULKAN_SEARCH_PATHS) {
    if (fs.existsSync(p)) {
      return p;
    }
  }
  return null;
}

const nativeBridge = require('./native_bridge');

class Doctor {
  constructor(statePath = null) {
    this.statePath = statePath || getDefaultCachePath();
    this.stageNames = [
      "Vulkan Loader Open",
      "Instance Creation",
      "Physical Device Enumeration",
      "Hardware GPU Selection",
      "Compute Queue Family Probe",
      "Logical Device Creation",
      "Buffer Allocation & Mapping",
      "SPIR-V Pipeline Compilation",
      "Compute Shader Dispatch",
      "Result Checksum Validation",
      "GGML MatMul Tensor Ops",
      "End-to-End Model Inference"
    ];
  }

  async runSelfTest(verbose = true) {
    // 1. If Native Addon is loaded, delegate directly to Native C ABI Doctor
    if (nativeBridge.isNativeLoaded()) {
      const diag = await nativeBridge.runDiagnostic({ verbose });
      const report = {
        schemaVersion: 2,
        overallSuccess: Boolean(diag.overallSuccess),
        diagnosticScope: diag.diagnosticScope || "compute",
        scopeSuccess: diag.scopeSuccess !== undefined ? Boolean(diag.scopeSuccess) : Boolean(diag.overallSuccess),
        computeCertified: Boolean(diag.computeCertified),
        modelCertified: Boolean(diag.modelCertified),
        capabilityStatus: diag.computeCertified ? "COMPUTE_CERTIFIED" : "COMPUTE_UNVERIFIED",
        verificationSource: "native_c_hal",
        verifiedAt: new Date().toISOString(),
        deviceName: diag.deviceName || "Unknown",
        driverVersion: diag.driverVersion || "Unknown",
        loaderPath: diag.loaderPath || "Unknown",
        passedStages: diag.passedStages,
        totalStages: diag.totalStages,
        totalElapsedMs: diag.totalElapsedMs,
        recommendedBackend: diag.recommendedBackend,
        exitCode: diag.exitCode,
        stages: []
      };
      return report;
    }

    // 2. Pure JS Probe when native addon is unavailable
    const t0 = performance.now();
    if (verbose) {
      console.log("\n============================================================");
      console.log("  AMEVA-Vulkan-Runtime (Node.js): 12-Stage Diagnostic Suite ");
      console.log("============================================================");
    }

    const stages = [];
    let passed = 0;
    let overallSuccess = false;
    let deviceName = "Unknown";
    let loaderPath = findVulkanLib();

    if (loaderPath) {
      stages.push({
        stageId: 0,
        stageName: `V0: ${this.stageNames[0]}`,
        result: "PASS",
        elapsedMs: 0.5,
        detailMessage: `Bound to: ${loaderPath}`
      });
      passed++;
      deviceName = "Android Vulkan ICD Driver Detected";

      // V1~V6: Probe native execution status honestly
      for (let i = 1; i <= 6; i++) {
        stages.push({
          stageId: i,
          stageName: `V${i}: ${this.stageNames[i]}`,
          result: "SKIP",
          elapsedMs: 0.0,
          detailMessage: "Native C HAL FFI (libameva_vulkan.so) binding required for hardware dispatch"
        });
      }

      for (let i = 7; i < 12; i++) {
        stages.push({
          stageId: i,
          stageName: `V${i}: ${this.stageNames[i]}`,
          result: "SKIP",
          elapsedMs: 0.0,
          detailMessage: "End-to-end shader verification unverified in standalone JS runtime"
        });
      }
      overallSuccess = false; // Pure JS without C FFI cannot certify Vulkan compute
    } else {
      stages.push({
        stageId: 0,
        stageName: `V0: ${this.stageNames[0]}`,
        result: "FAIL",
        elapsedMs: 0.2,
        detailMessage: "No Vulkan ICD library found on system"
      });
      for (let i = 1; i < 12; i++) {
        stages.push({
          stageId: i,
          stageName: `V${i}: ${this.stageNames[i]}`,
          result: "SKIP",
          elapsedMs: 0.0,
          detailMessage: "Skipped due to V0 failure"
        });
      }
      overallSuccess = false;
    }

    const totalElapsed = performance.now() - t0;
    const report = {
      schemaVersion: 2,
      overallSuccess,
      diagnosticScope: "loader",
      scopeSuccess: false,
      computeCertified: false,
      modelCertified: false,
      capabilityStatus: loaderPath ? "LOADER_PRESENT_UNVERIFIED" : "LOADER_NOT_FOUND",
      verificationSource: "pure_js_probe",
      verifiedAt: new Date().toISOString(),
      deviceName,
      driverVersion: "Vulkan 1.1+",
      loaderPath: loaderPath || "None",
      passedStages: passed,
      totalStages: 12,
      totalElapsedMs: totalElapsed,
      recommendedBackend: "cpu_neon",
      stages
    };

    if (verbose) {
      console.log("------------------------------------------------------------");
      console.log(`  Scorecard: ${passed}/12 Stages Passed | Time: ${totalElapsed.toFixed(2)} ms | Backend: ${report.recommendedBackend}`);
      console.log("============================================================\n");
    }

    // Strict Anti-Corruption: Pure JS probe MUST NOT overwrite valid native hardware certification state!
    try {
      const jsProbePath = path.join(path.dirname(this.statePath), 'vulkan_state_js_probe.json');
      fs.writeFileSync(jsProbePath, JSON.stringify(report, null, 2), 'utf-8');
    } catch (e) {}

    return report;
  }

  /**
   * Certificate Metadata Validation (Honest Pre-execution Filter)
   */
  validateCertificateMetadata(data) {
    if (!data || data.schemaVersion !== 2) return false;
    const fp = data.deviceFingerprint || data;
    const vendor = fp.vendorId !== undefined ? fp.vendorId : fp.vendor_id;
    if (!vendor || typeof vendor !== 'number' || vendor <= 0) return false;
    const devId = fp.deviceId !== undefined ? fp.deviceId : fp.device_id;
    if (devId === undefined || devId === null) return false;
    const devName = fp.deviceName || fp.device_name;
    if (!devName || devName === 'Unknown') return false;
    const lPath = fp.loaderPath || fp.loader_path;
    if (!lPath || typeof lPath !== 'string') return false;
    if (!fs.existsSync(lPath)) return false;
    try {
      fs.realpathSync(lPath);
    } catch {
      return false;
    }
    const currentLib = findVulkanLib();
    if (!currentLib) return false;
    return true;
  }

  // Alias for backward compatibility
  validateFingerprint(data) {
    return this.validateCertificateMetadata(data);
  }

  quickProbe() {
    // State cache verification (vulkan_state.json) is strictly enforced
    if (!fs.existsSync(this.statePath)) {
      return false;
    }
    try {
      const data = JSON.parse(fs.readFileSync(this.statePath, 'utf-8'));
      const now = Date.now();
      const verifiedTime = data.verifiedAt ? new Date(data.verifiedAt).getTime() : (data.timestamp ? data.timestamp * 1000 : 0);
      if (!verifiedTime || (now - verifiedTime) > MAX_CERT_AGE_MS) {
        return false;
      }
      if (data.overallSuccess !== true && data.overall_success !== true) {
        return false;
      }
      const isNativeC = data.verificationSource === "native_c_hal";
      if (!isNativeC) {
        return false;
      }
      const val = data.validation || data;
      const passedStages = val.passedStages !== undefined ? val.passedStages : val.passed_stages;
      const totalStages = val.totalStages !== undefined ? val.totalStages : val.total_stages;
      if (typeof passedStages !== 'number' || passedStages < 10) {
        return false;
      }
      if (typeof totalStages === 'number' && totalStages < 10) {
        return false;
      }

      // Strict compute scope certification:
      const isComputeCertified = Boolean(
        data.computeCertified === true ||
        (data.diagnosticScope === "compute" && (data.scopeSuccess === true || data.overallSuccess === true)) ||
        (!data.diagnosticScope && data.overallSuccess === true)
      );
      if (!isComputeCertified) {
        return false;
      }

      if (!this.validateCertificateMetadata(data)) {
        return false;
      }

      return true;
    } catch {
      return false;
    }
  }

  quickProbeDevice() {
    if (nativeBridge.isNativeLoaded()) {
      try {
        const directProbe = nativeBridge.quickProbe();
        if (directProbe && directProbe.available && directProbe.deviceName) {
          return directProbe.deviceName;
        }
      } catch (e) {}
    }
    if (fs.existsSync(this.statePath)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.statePath, 'utf-8'));
        const fp = data.deviceFingerprint || data;
        const name = fp.deviceName || data.deviceName;
        if (name && name !== 'Unknown') {
          return name;
        }
      } catch (e) {}
    }
    return null;
  }
}

module.exports = { Doctor };
