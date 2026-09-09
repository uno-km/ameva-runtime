const path = require('path');
const fs = require('fs');
const { PlatformNotSupportedError } = require('./errors');

let nativeModule = null;
let loadError = null;

const candidatePaths = [
  process.env.AMEVA_NATIVE_PATH,
  path.resolve(__dirname, '../../build/Release/ameva_native.node'),
  path.resolve(__dirname, '../../build/Debug/ameva_native.node'),
  path.resolve(__dirname, '../build/Release/ameva_native.node'),
  path.resolve(__dirname, './ameva_native.node')
].filter(Boolean);

for (const candidate of candidatePaths) {
  if (fs.existsSync(candidate)) {
    try {
      nativeModule = require(candidate);
      break;
    } catch (err) {
      loadError = err;
    }
  }
}

function ensureNative() {
  if (!nativeModule) {
    const detail = loadError ? ` (Cause: ${loadError.message})` : '';
    throw new PlatformNotSupportedError(
      `Native Ameva C ABI addon (ameva_native.node) is not loaded or not built for this platform${detail}. ` +
      `Zero-silent-fallback policy strictly enforced: manual build via 'node-gyp rebuild' is required.`
    );
  }
  return nativeModule;
}

function isNativeLoaded() {
  return nativeModule !== null;
}

function getBridgeVersion() {
  const mod = ensureNative();
  return mod.getBridgeVersion();
}

function getNativeAbiVersion() {
  const mod = ensureNative();
  return mod.getNativeAbiVersion();
}

function getDiagnosticResultStructSize() {
  const mod = ensureNative();
  return mod.getDiagnosticResultStructSize();
}

function getVersion() {
  const mod = ensureNative();
  return mod.getVersion();
}

function isTensorAligned(ne01, ne11) {
  const mod = ensureNative();
  return mod.isTensorAligned(ne01, ne11);
}

function isVulkanAvailable() {
  const mod = ensureNative();
  return mod.isVulkanAvailable();
}

function quickProbe() {
  const mod = ensureNative();
  return mod.quickProbe();
}

async function runDiagnostic(options = {}) {
  const mod = ensureNative();
  return await mod.runDiagnostic(options);
}

async function matmulF32(a, b, m, k, n) {
  const mod = ensureNative();
  return await mod.matmulF32(a, b, m, k, n);
}

async function matmulF32WithTelemetry(a, b, m, k, n) {
  const mod = ensureNative();
  return await mod.matmulF32WithTelemetry(a, b, m, k, n);
}

function getConcurrencyMetrics() {
  const mod = ensureNative();
  return mod.getConcurrencyMetrics();
}

function resetConcurrencyMetrics() {
  const mod = ensureNative();
  return mod.resetConcurrencyMetrics();
}

module.exports = {
  isNativeLoaded,
  getBridgeVersion,
  getNativeAbiVersion,
  getDiagnosticResultStructSize,
  getVersion,
  isTensorAligned,
  isVulkanAvailable,
  quickProbe,
  runDiagnostic,
  matmulF32,
  matmulF32WithTelemetry,
  getConcurrencyMetrics,
  resetConcurrencyMetrics,
  debug: {
    getConcurrencyMetrics,
    resetConcurrencyMetrics
  }
};

