'use strict';

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

/**
 * Common safe environment variables passed to subprocesses.
 * PATH is provided solely for internal tool execution by the child; it is NOT used to resolve the executable.
 */
const SAFE_BASE_ENV_KEYS = [
  'HOME',
  'TMPDIR',
  'TEMP',
  'TMP',
  'LANG',
  'LC_ALL',
  'TERM',
  'ANDROID_DATA',
  'ANDROID_ROOT',
  'PREFIX',
  'PATH'
];

/**
 * Dangerous variables that can manipulate dynamic linking or interpreter execution.
 * These are unconditionally blocked from user customEnv and parent inheritance.
 */
const BLOCKED_ENV_KEYS = [
  'LD_PRELOAD',
  'NODE_OPTIONS',
  'NODE_PATH',
  'BASH_ENV',
  'ENV',
  'IFS',
  'VK_LAYER_PATH'
];

/**
 * Standard system root prefixes permitted for runtime dynamic libraries and Vulkan ICD manifests.
 */
const DEFAULT_ALLOWED_RUNTIME_ROOTS = [
  '/system',
  '/vendor',
  '/data/data/com.termux/files/usr',
  '/usr',
  '/opt'
];

/**
 * Validates the executable path against strict security contracts:
 * - Absolute path only
 * - No NUL bytes
 * - Must be a regular file (not directory or socket)
 * - Windows: strictly .exe (rejects .cmd, .bat, .ps1)
 * - POSIX: executable bit (X_OK) required
 *
 * @param {string} executable Absolute path to executable
 * @returns {Promise<string>} Normalized absolute path
 */
async function validateExecutable(executable) {
  if (typeof executable !== 'string' || executable.trim() === '') {
    throw new TypeError('Executable path must be a non-empty string');
  }

  if (executable.includes('\0')) {
    throw new Error('Executable path contains forbidden NUL character');
  }

  if (!path.isAbsolute(executable)) {
    throw new Error(`Executable path must be absolute: ${executable}`);
  }

  const normalized = path.normalize(executable);

  if (process.platform === 'win32') {
    const ext = path.extname(normalized).toLowerCase();
    if (ext !== '.exe') {
      throw new Error(`On Windows, only direct .exe executables are permitted with shell:false. Got: '${ext}' for ${normalized}`);
    }
  }

  let stat;
  try {
    stat = await fs.promises.stat(normalized);
  } catch (err) {
    throw new Error(`Executable file not found: ${normalized} (${err.message})`);
  }

  if (!stat.isFile()) {
    throw new Error(`Executable path is not a regular file: ${normalized}`);
  }

  if (process.platform !== 'win32') {
    try {
      await fs.promises.access(normalized, fs.constants.X_OK);
    } catch (err) {
      throw new Error(`Executable lacks execution permission (X_OK): ${normalized}`);
    }
  }

  return normalized;
}

/**
 * Checks if a candidate path is contained within any of the allowed root directories
 * using path.relative to prevent directory traversal attacks.
 */
async function isUnderAllowedRoots(candidatePath, allowedRoots) {
  let resolvedCandidate = path.resolve(candidatePath);
  try {
    resolvedCandidate = await fs.promises.realpath(resolvedCandidate);
  } catch (_) {
    // If realpath fails (e.g. file doesn't exist yet), use resolved path
  }

  for (const root of allowedRoots) {
    let resolvedRoot = path.resolve(root);
    try {
      resolvedRoot = await fs.promises.realpath(resolvedRoot);
    } catch (_) {}

    const relative = path.relative(resolvedRoot, resolvedCandidate);
    const inside = relative === '' || (!relative.startsWith('..') && !path.isAbsolute(relative));
    if (inside) {
      return true;
    }
  }
  return false;
}

/**
 * Sanitizes and validates runtimeEnv entries (LD_LIBRARY_PATH, VK_ICD_FILENAMES).
 * Splits entries by path.delimiter and ensures each path is absolute and under an allowed root.
 */
async function validateAndBuildRuntimeEnv(runtimeEnv, allowedRoots = DEFAULT_ALLOWED_RUNTIME_ROOTS) {
  const result = {};
  if (!runtimeEnv || typeof runtimeEnv !== 'object') {
    return result;
  }

  const allowedVars = ['LD_LIBRARY_PATH', 'VK_ICD_FILENAMES'];

  for (const varName of allowedVars) {
    if (varName in runtimeEnv) {
      const val = runtimeEnv[varName];
      if (typeof val !== 'string') {
        throw new TypeError(`runtimeEnv.${varName} must be a string`);
      }
      if (val.includes('\0')) {
        throw new Error(`runtimeEnv.${varName} contains forbidden NUL character`);
      }

      const entries = val.split(path.delimiter);
      const validatedEntries = [];

      for (const rawEntry of entries) {
        const entry = rawEntry.trim();
        if (entry === '') {
          throw new Error(`runtimeEnv.${varName} contains empty path entry`);
        }
        if (!path.isAbsolute(entry)) {
          throw new Error(`runtimeEnv.${varName} entry must be an absolute path: ${entry}`);
        }

        // Check root containment if on POSIX (on Windows, driver paths may reside in C:\Windows\System32)
        if (process.platform !== 'win32') {
          const allowed = await isUnderAllowedRoots(entry, allowedRoots);
          if (!allowed) {
            throw new Error(`runtimeEnv.${varName} entry '${entry}' is not located under allowed root directories`);
          }
        }

        validatedEntries.push(path.normalize(entry));
      }

      result[varName] = validatedEntries.join(path.delimiter);
    }
  }

  return result;
}

/**
 * Builds a strictly isolated environment for the subprocess.
 */
async function buildSubprocessEnv(customEnv, runtimeEnv, allowedRoots) {
  const finalEnv = {};

  // 1. Inherit safe base keys from parent process
  for (const key of SAFE_BASE_ENV_KEYS) {
    if (key in process.env && process.env[key] !== undefined) {
      finalEnv[key] = process.env[key];
    }
  }

  // 2. Overlay custom user env if provided, strictly filtering out blocked keys
  if (customEnv && typeof customEnv === 'object') {
    for (const [key, val] of Object.entries(customEnv)) {
      if (BLOCKED_ENV_KEYS.includes(key)) {
        continue; // Unconditionally block dangerous keys
      }
      // LD_LIBRARY_PATH and VK_ICD_FILENAMES are forbidden in general customEnv
      if (key === 'LD_LIBRARY_PATH' || key === 'VK_ICD_FILENAMES') {
        continue;
      }
      if (typeof val === 'string' && !val.includes('\0')) {
        finalEnv[key] = val;
      }
    }
  }

  // 3. Process authorized runtimeEnv
  const safeRuntime = await validateAndBuildRuntimeEnv(runtimeEnv, allowedRoots);
  Object.assign(finalEnv, safeRuntime);

  // 4. Expose designated telemetry file descriptor
  finalEnv.AMEVA_TELEMETRY_FD = '3';

  return finalEnv;
}

/**
 * Appends chunk to buffer list adhering to exact byte limit contract:
 * - bytes == limit: permitted (exceeded: false)
 * - bytes > limit: exceeded (exceeded: true)
 * - Slices are copied via Buffer.from to release reference to larger V8 slabs.
 */
function appendBoundedChunk(chunk, chunks, currentBytes, limit) {
  const remaining = limit - currentBytes;

  if (remaining <= 0) {
    return {
      bytes: currentBytes,
      exceeded: chunk.length > 0
    };
  }

  if (chunk.length > remaining) {
    chunks.push(Buffer.from(chunk.subarray(0, remaining)));
    return {
      bytes: limit,
      exceeded: true
    };
  }

  chunks.push(Buffer.from(chunk));
  return {
    bytes: currentBytes + chunk.length,
    exceeded: false
  };
}

/**
 * Validates the telemetry device string:
 * Must be string, non-null, no NUL bytes, no control characters (0-31, 127), <= 256 UTF-8 bytes.
 */
function validateTelemetryDevice(device) {
  if (typeof device !== 'string') {
    return null;
  }
  if (device.includes('\0')) {
    return null;
  }
  // Check control chars
  for (let i = 0; i < device.length; i++) {
    const code = device.charCodeAt(i);
    if ((code >= 0 && code <= 31) || code === 127) {
      return null;
    }
  }
  const byteLen = Buffer.byteLength(device, 'utf8');
  if (byteLen > 256) {
    return null;
  }
  return device;
}

/**
 * Executes a subprocess with strict resource bounds, signal lifecycle management,
 * and structured telemetry verification over dedicated FD 3.
 *
 * @param {Object} options Subprocess execution options
 * @param {string} options.executable Absolute path to executable
 * @param {string[]} [options.args] Argument array
 * @param {Object} [options.env] Custom environment variables
 * @param {Object} [options.runtimeEnv] Authorized loader/ICD variables
 * @param {string[]} [options.allowedRuntimeRoots] Allowed directory roots for runtimeEnv
 * @param {string} [options.cwd] Working directory (defaults to process.cwd())
 * @param {number} [options.timeoutMs=30000] Timeout before SIGTERM (0 for indefinite)
 * @param {number} [options.gracePeriodMs=2000] Grace period after SIGTERM before SIGKILL
 * @param {number} [options.maxStdoutBytes=10485760] Maximum stdout buffer size (10MB)
 * @param {number} [options.maxStderrBytes=10485760] Maximum stderr buffer size (10MB)
 * @param {number} [options.maxTelemetryBytes=1048576] Maximum FD 3 telemetry buffer size (1MB)
 * @param {number} [options.maxTelemetryLineBytes=16384] Maximum single telemetry line size (16KB)
 * @param {number} [options.maxTelemetryMessages=1024] Maximum telemetry message count
 * @param {AbortSignal} [options.signal] External abort signal
 * @param {string|null} [options.backendRequested=null] Requested acceleration backend
 * @returns {Promise<Object>} Execution result object
 */
async function executeSubprocess(options) {
  if (!options || typeof options !== 'object') {
    throw new TypeError('Options must be a non-null object');
  }

  // 1. Pre-execution AbortSignal check
  if (options.signal && options.signal.aborted) {
    const err = new Error('Execution aborted before spawn');
    err.name = 'AbortError';
    throw err;
  }

  // 2. Validate executable
  const validatedExecutable = await validateExecutable(options.executable);

  // 3. Validate arguments
  const args = options.args || [];
  if (!Array.isArray(args)) {
    throw new TypeError('Options.args must be an Array of strings');
  }
  for (let i = 0; i < args.length; i++) {
    if (typeof args[i] !== 'string') {
      throw new TypeError(`Argument at index ${i} is not a string: ${typeof args[i]}`);
    }
  }

  // 4. Configuration parameters
  const timeoutMs = typeof options.timeoutMs === 'number' && options.timeoutMs >= 0 ? options.timeoutMs : 30000;
  const gracePeriodMs = typeof options.gracePeriodMs === 'number' && options.gracePeriodMs >= 0 ? options.gracePeriodMs : 2000;
  const maxStdoutBytes = typeof options.maxStdoutBytes === 'number' && options.maxStdoutBytes > 0 ? options.maxStdoutBytes : 10 * 1024 * 1024;
  const maxStderrBytes = typeof options.maxStderrBytes === 'number' && options.maxStderrBytes > 0 ? options.maxStderrBytes : 10 * 1024 * 1024;
  const maxTelemetryBytes = typeof options.maxTelemetryBytes === 'number' && options.maxTelemetryBytes > 0 ? options.maxTelemetryBytes : 1024 * 1024;
  const maxTelemetryLineBytes = typeof options.maxTelemetryLineBytes === 'number' && options.maxTelemetryLineBytes > 0 ? options.maxTelemetryLineBytes : 16 * 1024;
  const maxTelemetryMessages = typeof options.maxTelemetryMessages === 'number' && options.maxTelemetryMessages > 0 ? options.maxTelemetryMessages : 1024;
  const cwd = options.cwd && typeof options.cwd === 'string' ? options.cwd : process.cwd();
  const backendRequested = typeof options.backendRequested === 'string' ? options.backendRequested : null;

  const env = await buildSubprocessEnv(options.env, options.runtimeEnv, options.allowedRuntimeRoots);

  return new Promise((resolve) => {
    // Buffers and tracking
    const stdoutChunks = [];
    let stdoutBytes = 0;
    let stdoutTruncated = false;

    const stderrChunks = [];
    let stderrBytes = 0;
    let stderrTruncated = false;

    let telemetryBytes = 0;
    let telemetryTruncated = false;
    let telemetryRemainder = '';
    let telemetryMessageCount = 0;
    let telemetryLineNumber = 0;

    // Structured Telemetry State Machine
    // UNSET | CONFIRMED | CONFLICTED (Permanent locking)
    let backendTelemetryState = 'UNSET';
    let backendConfirmed = null;
    let backendDevice = null;
    let verificationSource = 'not_reported';
    const telemetryErrors = [];
    const metrics = {};
    const backendHints = [];

    // State Machine Flags
    let spawned = false;
    let terminationStarted = false;
    let finalized = false;
    let terminationReason = null;
    const secondaryReasons = [];
    let termSignalAttempted = false;
    let termSignalDelivered = false;
    let killSignalAttempted = false;
    let killSignalDelivered = false;
    let timedOut = false;
    let aborted = false;
    let spawnErrorCode = null;

    let child = null;
    let timeoutTimer = null;
    let graceTimer = null;
    const startTime = Date.now();

    /**
     * First-cause-wins termination requester
     */
    function requestTermination(reason) {
      if (!terminationStarted) {
        terminationStarted = true;
        terminationReason = reason;

        if (child && !termSignalAttempted) {
          termSignalAttempted = true;
          try {
            termSignalDelivered = Boolean(child.kill('SIGTERM'));
          } catch (_) {
            termSignalDelivered = false;
          }
        }

        graceTimer = setTimeout(() => {
          if (!finalized && !killSignalAttempted && child) {
            killSignalAttempted = true;
            try {
              killSignalDelivered = Boolean(child.kill('SIGKILL'));
            } catch (_) {
              killSignalDelivered = false;
            }
          }
        }, gracePeriodMs);
      } else {
        if (reason !== terminationReason && !secondaryReasons.includes(reason)) {
          secondaryReasons.push(reason);
        }
      }
    }


    /**
     * Parses a single line from FD 3 streaming JSONL
     */
    function parseTelemetryLine(lineStr) {
      telemetryLineNumber++;
      const trimmed = lineStr.trim();
      if (!trimmed) return;

      if (telemetryMessageCount >= maxTelemetryMessages) {
        requestTermination('OUTPUT_LIMIT_EXCEEDED');
        return;
      }
      telemetryMessageCount++;

      let parsed;
      try {
        parsed = JSON.parse(trimmed);
      } catch (err) {
        telemetryErrors.push({ code: 'INVALID_JSON', lineNumber: telemetryLineNumber, error: err.message });
        return;
      }

      if (!parsed || typeof parsed !== 'object') {
        telemetryErrors.push({ code: 'MALFORMED_MESSAGE', lineNumber: telemetryLineNumber });
        return;
      }


      // Backend affirmation message
      if (parsed.type === 'ameva.execution.backend') {
        if (parsed.schemaVersion !== 1) {
          telemetryErrors.push({ code: 'UNSUPPORTED_SCHEMA_VERSION', lineNumber: telemetryLineNumber });
          return;
        }

        const validBackends = ['vulkan', 'cpu_neon', 'cpu_reference'];
        if (!validBackends.includes(parsed.backend)) {
          telemetryErrors.push({ code: 'INVALID_BACKEND_VALUE', lineNumber: telemetryLineNumber });
          return;
        }

        const validatedDev = validateTelemetryDevice(parsed.device);

        if (backendTelemetryState === 'CONFLICTED') {
          // Permanently locked in CONFLICTED state
          return;
        }

        if (backendTelemetryState === 'UNSET') {
          backendConfirmed = parsed.backend;
          backendDevice = validatedDev;
          verificationSource = 'engine_structured_telemetry';
          backendTelemetryState = 'CONFIRMED';
          return;
        }

        if (backendConfirmed !== parsed.backend) {
          // Conflict detected -> Permanent lockout
          backendConfirmed = null;
          backendDevice = null;
          verificationSource = 'telemetry_conflict';
          backendTelemetryState = 'CONFLICTED';
        }
      }

      // Execution metrics message
      else if (parsed.type === 'ameva.execution.metrics') {
        if (parsed.schemaVersion === 1) {
          if (typeof parsed.processingDurationMs === 'number' && Number.isFinite(parsed.processingDurationMs) && parsed.processingDurationMs > 0) {
            metrics.processingDurationMs = parsed.processingDurationMs;
          }
          if (typeof parsed.inputDurationMs === 'number' && Number.isFinite(parsed.inputDurationMs) && parsed.inputDurationMs > 0) {
            metrics.inputDurationMs = parsed.inputDurationMs;
          }
          if (typeof parsed.tokensPerSecond === 'number' && Number.isFinite(parsed.tokensPerSecond) && parsed.tokensPerSecond > 0) {
            metrics.tokensPerSecond = parsed.tokensPerSecond;
          }
          // Calculate RTF strictly if both durations are valid
          if (metrics.processingDurationMs && metrics.inputDurationMs) {
            metrics.rtf = metrics.processingDurationMs / metrics.inputDurationMs;
          }
        }
      }
    }

    /**
     * Idempotent finalizer that cleans up timers and settles the Promise exactly once.
     */
    function finalize(exitCode, signal, spawnErr) {
      if (finalized) return;
      finalized = true;

      // 1. Clear timers
      if (timeoutTimer) {
        clearTimeout(timeoutTimer);
        timeoutTimer = null;
      }
      if (graceTimer) {
        clearTimeout(graceTimer);
        graceTimer = null;
      }

      // 2. Remove abort listener
      if (options.signal && onAbort) {
        try {
          options.signal.removeEventListener('abort', onAbort);
        } catch (_) {}
      }

      // 3. Process remaining telemetry data if any
      if (telemetryRemainder && telemetryRemainder.trim() && !telemetryTruncated) {
        parseTelemetryLine(telemetryRemainder);
        telemetryRemainder = '';
      }

      const durationMs = Date.now() - startTime;
      const stdout = Buffer.concat(stdoutChunks).toString('utf8');
      const stderr = Buffer.concat(stderrChunks).toString('utf8');

      // 4. Extract informal backend diagnostic hints from stdout/stderr without confirming backend
      const combinedLogs = stdout + '\n' + stderr;
      if (/ggml_vulkan:\s*GPU/i.test(combinedLogs) || /backend:\s*vulkan/i.test(combinedLogs)) {
        backendHints.push('vulkan_detected_in_log');
      }
      if (/ggml_neon:\s*CPU/i.test(combinedLogs) || /backend:\s*cpu_neon/i.test(combinedLogs)) {
        backendHints.push('neon_detected_in_log');
      }

      // 5. Determine primary terminationReason if not already established
      let finalTerminationReason = terminationReason;
      if (!finalTerminationReason) {
        if (spawnErr) {
          finalTerminationReason = 'SPAWN_ERROR';
        } else if (aborted) {
          finalTerminationReason = 'ABORTED';
        } else if (timedOut) {
          finalTerminationReason = 'TIMEOUT';
        } else if (exitCode !== 0 && exitCode !== null) {
          finalTerminationReason = 'EXIT_NONZERO';
        } else {
          finalTerminationReason = 'COMPLETED';
        }
      }

      const result = {
        spawned,
        pid: child ? child.pid || null : null,
        exitCode: exitCode !== undefined ? exitCode : null,
        signal: signal || null,
        stdout,
        stderr,
        durationMs,
        terminationReason: finalTerminationReason,
        secondaryReasons: [...secondaryReasons],
        timedOut,
        aborted,
        stdoutTruncated,
        stderrTruncated,
        spawnErrorCode: spawnErr ? (spawnErr.code || spawnErr.name || 'UNKNOWN') : spawnErrorCode,
        termSignalAttempted,
        termSignalDelivered,
        killSignalAttempted,
        killSignalDelivered,
        termSignalSent: termSignalDelivered,
        killSignalSent: killSignalDelivered,
        backendRequested,
        backendConfirmed,
        backendDevice,
        verificationSource,
        telemetryErrors,
        metrics,
        backendHints
      };

      resolve(result);
    }

    // AbortSignal listener
    function onAbort() {
      aborted = true;
      requestTermination('ABORTED');
    }

    if (options.signal) {
      options.signal.addEventListener('abort', onAbort, { once: true });
    }

    // Timeout scheduler
    if (timeoutMs > 0) {
      timeoutTimer = setTimeout(() => {
        timedOut = true;
        requestTermination('TIMEOUT');
      }, timeoutMs);
    }

    // Spawn execution
    try {
      child = spawn(validatedExecutable, args, {
        cwd,
        env,
        shell: false,
        stdio: ['ignore', 'pipe', 'pipe', 'pipe']
      });
    } catch (err) {
      finalize(null, null, err);
      return;
    }

    // Lifecycle events
    child.once('spawn', () => {
      spawned = true;
    });

    child.once('error', (err) => {
      spawnErrorCode = err.code || err.name || 'UNKNOWN';
      requestTermination('SPAWN_ERROR');
      finalize(null, null, err);
    });

    // Stdout handler
    if (child.stdout) {
      child.stdout.on('data', (chunk) => {
        const res = appendBoundedChunk(chunk, stdoutChunks, stdoutBytes, maxStdoutBytes);
        stdoutBytes = res.bytes;
        if (res.exceeded) {
          stdoutTruncated = true;
          requestTermination('OUTPUT_LIMIT_EXCEEDED');
        }
      });
    }

    // Stderr handler
    if (child.stderr) {
      child.stderr.on('data', (chunk) => {
        const res = appendBoundedChunk(chunk, stderrChunks, stderrBytes, maxStderrBytes);
        stderrBytes = res.bytes;
        if (res.exceeded) {
          stderrTruncated = true;
          requestTermination('OUTPUT_LIMIT_EXCEEDED');
        }
      });
    }

    // Telemetry (FD 3) streaming handler
    if (child.stdio && child.stdio[3]) {
      const telemetryStream = child.stdio[3];

      telemetryStream.on('data', (chunk) => {
        telemetryBytes += chunk.length;
        if (telemetryBytes > maxTelemetryBytes) {
          telemetryTruncated = true;
          requestTermination('OUTPUT_LIMIT_EXCEEDED');
          return;
        }

        const chunkStr = chunk.toString('utf8');
        const combined = telemetryRemainder + chunkStr;
        const lines = combined.split('\n');
        telemetryRemainder = lines.pop(); // Incomplete last line

        for (const line of lines) {
          if (Buffer.byteLength(line, 'utf8') > maxTelemetryLineBytes) {
            telemetryErrors.push({ code: 'LINE_LENGTH_EXCEEDED', lineNumber: telemetryLineNumber + 1 });
            continue;
          }
          parseTelemetryLine(line);
        }
      });
    }

    // Intermediate exit state tracking
    let exitedCode = null;
    let exitedSignal = null;

    child.once('exit', (code, signal) => {
      exitedCode = code;
      exitedSignal = signal;
    });

    // Process close handler guarantees stdout/stderr/FD 3 stream termination
    child.once('close', (closeCode, closeSignal) => {
      const finalCode = closeCode !== null && closeCode !== undefined ? closeCode : exitedCode;
      const finalSig = closeSignal || exitedSignal;
      finalize(finalCode, finalSig, null);
    });
  });
}


module.exports = {
  validateExecutable,
  executeSubprocess,
  appendBoundedChunk,
  validateTelemetryDevice,
  SAFE_BASE_ENV_KEYS,
  BLOCKED_ENV_KEYS
};
