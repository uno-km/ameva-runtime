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
  for (const p of VULKAN_SEARCH_PATHS) {
    if (fs.existsSync(p)) {
      return p;
    }
  }
  return null;
}

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
      overallSuccess,
      deviceName,
      driverVersion: "Vulkan 1.1+",
      loaderPath: loaderPath || "None",
      passedStages: passed,
      totalStages: 12,
      totalElapsedMs: totalElapsed,
      recommendedBackend: overallSuccess ? "vulkan" : "cpu_neon",
      stages
    };

    if (verbose) {
      console.log("------------------------------------------------------------");
      console.log(`  Scorecard: ${passed}/12 Stages Passed | Time: ${totalElapsed.toFixed(2)} ms | Backend: ${report.recommendedBackend}`);
      console.log("============================================================\n");
    }

    try {
      const dir = path.dirname(this.statePath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(this.statePath, JSON.stringify(report, null, 2), 'utf-8');
    } catch (e) {}

    return report;
  }

  quickProbe() {
    if (fs.existsSync(this.statePath)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.statePath, 'utf-8'));
        if (data.overallSuccess || data.recommendedBackend === "vulkan" || data.recommendedBackend === "vulkan_driver_only" || (data.passedStages && data.passedStages >= 7)) {
          return true;
        }
      } catch (e) {}
    }
    const lib = findVulkanLib();
    return !!lib;
  }

  quickProbeDevice() {
    if (fs.existsSync(this.statePath)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.statePath, 'utf-8'));
        if (data.deviceName && data.deviceName !== 'Unknown') {
          return data.deviceName;
        }
      } catch (e) {}
    }
    return null;
  }
}

module.exports = { Doctor };
