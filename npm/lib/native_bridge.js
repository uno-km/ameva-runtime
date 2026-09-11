const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
const { PlatformNotSupportedError } = require('./errors');

let nativeModule = null;
let loadError = null;
let addonInfo = {
  loaded: false,
  addonPath: null,
  errorCode: "NATIVE_ADDON_UNAVAILABLE",
  errorMessage: `No native addon is available for ${process.platform}-${process.arch}.`
};

function checkTermuxEnvironment() {
  if (process.platform !== 'android' || process.arch !== 'arm64') {
    return { isTermux: false, compatible: true };
  }
  const termuxPrefix = process.env.PREFIX || '/data/data/com.termux/files/usr';
  const isTermuxPath = typeof termuxPrefix === 'string' && termuxPrefix.includes('com.termux');
  const libcppPath = path.join(termuxPrefix, 'lib', 'libc++_shared.so');
  let hasLibCpp = false;
  try {
    hasLibCpp = fs.existsSync(libcppPath);
  } catch (_) {
    hasLibCpp = false;
  }
  if (!isTermuxPath || !hasLibCpp) {
    return {
      isTermux: false,
      compatible: false,
      errorCode: 'TERMUX_RUNTIME_DEPENDENCY_MISSING',
      errorMessage: `Official Termux runtime dependency missing: $PREFIX/lib/libc++_shared.so was not found at '${libcppPath}'. ` +
        `This prebuild is strictly targeted for Official Termux on Android arm64 and is not compatible with generic Android or non-Termux environments.`
    };
  }
  return { isTermux: true, compatible: true, prefix: termuxPrefix, libcppPath };
}

const termuxCheck = checkTermuxEnvironment();
if (process.platform === 'android' && process.arch === 'arm64' && !termuxCheck.compatible) {
  addonInfo = {
    loaded: false,
    addonPath: null,
    errorCode: termuxCheck.errorCode,
    errorMessage: termuxCheck.errorMessage
  };
  loadError = new Error(termuxCheck.errorMessage);
  loadError.code = termuxCheck.errorCode;
} else {
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
    if (process.env.NODE_ENV === 'production') {
      addonInfo.errorCode = 'NATIVE_OVERRIDE_FORBIDDEN_IN_PRODUCTION';
      addonInfo.errorMessage = 'AMEVA_ALLOW_NATIVE_OVERRIDE is strictly forbidden in production environments.';
    } else {
      const envPath = process.env.AMEVA_NATIVE_PATH;
      if (
        typeof envPath === 'string' &&
        !envPath.includes('\0') &&
        path.isAbsolute(envPath) &&
        envPath.endsWith('.node')
      ) {
        try {
          if (fs.existsSync(envPath)) {
            const lstat = fs.lstatSync(envPath);
            if (lstat.isSymbolicLink()) {
              addonInfo.errorCode = 'NATIVE_OVERRIDE_SYMLINK_REJECTED';
              addonInfo.errorMessage = `Native override path cannot be a symbolic link: ${envPath}`;
            } else {
              const realPath = fs.realpathSync(envPath);
              const realStat = fs.statSync(realPath);
              if (!realStat.isFile()) {
                addonInfo.errorCode = 'NATIVE_OVERRIDE_NOT_REGULAR_FILE';
                addonInfo.errorMessage = `Native override path must be a regular file: ${realPath}`;
              } else if (process.platform !== 'win32' && ((realStat.mode & 0o002) !== 0)) {
                addonInfo.errorCode = 'NATIVE_OVERRIDE_WORLD_WRITABLE';
                addonInfo.errorMessage = `Native override file is world-writable and insecure: ${realPath}`;
              } else {
                const parentDir = path.dirname(realPath);
                const parentStat = fs.statSync(parentDir);
                if (process.platform !== 'win32' && ((parentStat.mode & 0o002) !== 0)) {
                  addonInfo.errorCode = 'NATIVE_OVERRIDE_PARENT_WORLD_WRITABLE';
                  addonInfo.errorMessage = `Parent directory of native override is world-writable: ${parentDir}`;
                } else {
                  // Strict cryptographic SHA-256 verification
                  const expectedSha256 = (process.env.AMEVA_NATIVE_SHA256 || '').trim().toLowerCase();
                  if (!expectedSha256 || expectedSha256.length !== 64) {
                    addonInfo.errorCode = 'NATIVE_OVERRIDE_MISSING_CHECKSUM';
                    addonInfo.errorMessage = 'AMEVA_NATIVE_SHA256 (64-char hex) is required when using AMEVA_ALLOW_NATIVE_OVERRIDE=1.';
                  } else {
                    const fileBytes = fs.readFileSync(realPath);
                    const actualSha256 = crypto.createHash('sha256').update(fileBytes).digest('hex');
                    if (actualSha256 !== expectedSha256) {
                      addonInfo.errorCode = 'NATIVE_OVERRIDE_CHECKSUM_MISMATCH';
                      addonInfo.errorMessage = `Native override SHA-256 mismatch: expected ${expectedSha256}, got ${actualSha256}`;
                    } else {
                      candidates.unshift({ source: 'environment_override', path: realPath });
                    }
                  }
                }
              }
            }
          }
        } catch (err) {
          addonInfo.errorCode = 'NATIVE_OVERRIDE_ERROR';
          addonInfo.errorMessage = `Error inspecting native override: ${err.message}`;
        }
      }
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
          path: candidate.path,
          termuxRuntime: termuxCheck.isTermux ? {
            prefix: termuxCheck.prefix,
            libcxx: termuxCheck.libcppPath
          } : undefined
        };
        loadError = null;
        break;
      } catch (err) {
        loadError = err;
      }
    }
  }
}

function ensureNative() {
  if (!nativeModule) {
    const detail = loadError ? ` (Cause: ${loadError.message})` : '';
    const code = addonInfo.errorCode || "NATIVE_ADDON_UNAVAILABLE";
    const err = new PlatformNotSupportedError(
      `Native Ameva C ABI addon (ameva_native.node) is not loaded or not available for ${process.platform}-${process.arch}${detail}. ` +
      `Zero-silent-fallback policy strictly enforced: prebuilt binary or manual build via 'node-gyp rebuild' is required.`
    );
    err.code = code;
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

