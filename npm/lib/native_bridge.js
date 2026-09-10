const path = require('path');
const fs = require('fs');
const { PlatformNotSupportedError } = require('./errors');

let nativeModule = null;
let loadError = null;
let addonInfo = {
  loaded: false,
  addonPath: null,
  errorCode: "NATIVE_ADDON_UNAVAILABLE",
  errorMessage: `No native addon is available for ${process.platform}-${process.arch}.`
};

// 1. Packaged prebuilds take first priority
const candidates = [
  { source: 'prebuilt', path: path.resolve(__dirname, '../prebuilds', `${process.platform}-${process.arch}`, 'ameva_native.node') },
  { source: 'build_release', path: path.resolve(__dirname, '../build/Release/ameva_native.node') },
  { source: 'dev_build_release', path: path.resolve(__dirname, '../../build/Release/ameva_native.node') },
  { source: 'dev_build_debug', path: path.resolve(__dirname, '../../build/Debug/ameva_native.node') },
  { source: 'local', path: path.resolve(__dirname, './ameva_native.node') }
];

// 2. Strict opt-in environment override validation (requires AMEVA_ALLOW_NATIVE_OVERRIDE=1)
if (process.env.AMEVA_ALLOW_NATIVE_OVERRIDE === '1' && process.env.AMEVA_NATIVE_PATH) {
  const envPath = process.env.AMEVA_NATIVE_PATH;
  if (
    typeof envPath === 'string' &&
    !envPath.includes('\0') &&
    path.isAbsolute(envPath) &&
    envPath.endsWith('.node')
  ) {
    try {
      if (fs.existsSync(envPath) && fs.statSync(envPath).isFile()) {
        candidates.unshift({ source: 'environment_override', path: envPath });
      }
    } catch (_) {}
  }
}

for (const candidate of candidates) {
  if (candidate.path && fs.existsSync(candidate.path)) {
    try {
      nativeModule = require(candidate.path);
      addonInfo = {
        loaded: true,
        source: candidate.source,
        platform: process.platform,
        arch: process.arch,
        path: candidate.path
      };
      loadError = null;
      break;
    } catch (err) {
      loadError = err;
    }
  }
}

function ensureNative() {
  if (!nativeModule) {
    const detail = loadError ? ` (Cause: ${loadError.message})` : '';
    const err = new PlatformNotSupportedError(
      `Native Ameva C ABI addon (ameva_native.node) is not loaded or not available for ${process.platform}-${process.arch}${detail}. ` +
      `Zero-silent-fallback policy strictly enforced: prebuilt binary or manual build via 'node-gyp rebuild' is required.`
    );
    err.code = "NATIVE_ADDON_UNAVAILABLE";
    throw err;
  }
  return nativeModule;
}

function isNativeLoaded() {
  return nativeModule !== null;
}

function getNativeAddonInfo() {
  return { ...addonInfo };
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
  getNativeAddonInfo,
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

