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

const { PlatformNotSupportedError } = require('./errors');
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
    if (!nativeBridge.isNativeLoaded()) {
      const info = nativeBridge.getNativeAddonInfo();
      const err = new PlatformNotSupportedError(
        `Native Ameva C ABI addon is not available for ${process.platform}-${process.arch}. ` +
        `Hardware diagnostic requires verified native driver bindings. (${info.errorMessage || info.errorCode})`
      );
      err.code = info.errorCode || "NATIVE_ADDON_UNAVAILABLE";
      throw err;
    }

    // Delegate directly to Native C ABI Doctor
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
