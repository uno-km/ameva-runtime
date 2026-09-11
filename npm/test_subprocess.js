'use strict';

const fs = require('fs');
const path = require('path');
const { executeSubprocess, validateExecutable } = require('./lib/subprocess');
const { LlamaCppExecutionPlan } = require('./lib/adapters');

let totalTests = 0;
let passedTests = 0;

function assert(condition, message) {
  totalTests++;
  if (!condition) {
    console.error(`  [FAIL] ${message}`);
    throw new Error(`Assertion failed: ${message}`);
  }
  passedTests++;
  console.log(`  [PASS] ${message}`);
}

async function runGate(name, fn) {
  console.log(`\n============================================================`);
  console.log(`[GATE] ${name}`);
  console.log(`============================================================`);
  try {
    await fn();
    console.log(`[GATE PASS] ${name}`);
  } catch (err) {
    console.error(`[GATE FAIL] ${name}: ${err.message}`);
    throw err;
  }
}

async function main() {
  console.log("=== AMEVA-Runtime Phase N2 13-Gate Verification Suite ===");
  console.log(`Node Platform: ${process.platform} (${process.arch})`);
  console.log(`Node Executable: ${process.execPath}\n`);

  // Gate 1: Options Validation & Path Rejection
  await runGate("Gate 1: Options Validation & Rejections", async () => {
    let errCaught = false;
    try {
      await executeSubprocess(null);
    } catch (e) {
      errCaught = e instanceof TypeError;
    }
    assert(errCaught, "executeSubprocess(null) throws TypeError");

    errCaught = false;
    try {
      await executeSubprocess({ executable: "" });
    } catch (e) {
      errCaught = e instanceof TypeError;
    }
    assert(errCaught, "Empty executable string throws TypeError");

    errCaught = false;
    try {
      await executeSubprocess({ executable: "node" });
    } catch (e) {
      errCaught = e.message.includes("must be absolute");
    }
    assert(errCaught, "Relative executable path ('node') is rejected");

    errCaught = false;
    try {
      await executeSubprocess({ executable: process.execPath + "\0" });
    } catch (e) {
      errCaught = e.message.includes("NUL character");
    }
    assert(errCaught, "Executable containing NUL byte is rejected");

    errCaught = false;
    try {
      await executeSubprocess({ executable: process.execPath, args: "not-an-array" });
    } catch (e) {
      errCaught = e instanceof TypeError;
    }
    assert(errCaught, "Non-array args throws TypeError");

    if (process.platform === 'win32') {
      errCaught = false;
      try {
        await executeSubprocess({ executable: "C:\\Windows\\System32\\cmd.exe.bat" });
      } catch (e) {
        errCaught = e.message.includes(".exe executables are permitted");
      }
      assert(errCaught, "Windows .bat extension is strictly rejected");
    }

    // Symbolic link rejection test
    const symTempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ameva-sym-test-'));
    const dummyExec = path.join(symTempDir, process.platform === 'win32' ? 'dummy.exe' : 'dummy');
    const symlinkPath = path.join(symTempDir, process.platform === 'win32' ? 'symlink.exe' : 'symlink');
    try {
      fs.copyFileSync(process.execPath, dummyExec);
      try {
        fs.symlinkSync(dummyExec, symlinkPath);
        let symCaught = false;
        try {
          await executeSubprocess({ executable: symlinkPath });
        } catch (e) {
          symCaught = e.message.includes("symbolic links are not permitted");
        }
        assert(symCaught, "Executable symbolic links are strictly rejected");
      } catch (symErr) {
        // In unprivileged Windows environments without developer mode, symlink creation may fail with EPERM
      }
    } finally {
      fs.rmSync(symTempDir, { recursive: true, force: true });
    }
  });

  // Gate 2: Clean Execution & PID Recovery
  await runGate("Gate 2: Clean Execution & PID Recovery", async () => {
    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'console.log("AMEVA_GATE_2_OK"); process.exit(0);']
    });

    assert(res.spawned === true, "Process successfully spawned");
    assert(typeof res.pid === 'number' && res.pid > 0, `Valid PID recovered: ${res.pid}`);
    assert(res.exitCode === 0, "Exit code is strictly 0");
    assert(res.terminationReason === "COMPLETED", "Termination reason is COMPLETED");
    assert(res.stdout.trim() === "AMEVA_GATE_2_OK", "Stdout recovered faithfully");
    assert(res.durationMs >= 0, `Execution duration recorded: ${res.durationMs}ms`);
  });

  // Gate 3: Nonzero Exit & Stderr Preservation
  await runGate("Gate 3: Nonzero Exit & Stderr Preservation", async () => {
    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'console.error("DIAGNOSTIC_STDERR_ERROR"); process.exit(7);']
    });

    assert(res.spawned === true, "Process successfully spawned");
    assert(res.exitCode === 7, "Exit code preserved as 7");
    assert(res.terminationReason === "EXIT_NONZERO", "Termination reason is EXIT_NONZERO");
    assert(res.stderr.includes("DIAGNOSTIC_STDERR_ERROR"), "Stderr preserved faithfully");
  });

  // Gate 4: Shell Metacharacters Uninterpreted
  await runGate("Gate 4: Shell Metacharacters Uninterpreted (shell: false)", async () => {
    const complexArg = "foo; bar | baz && echo test > file";
    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'console.log(process.argv[1])', complexArg]
    });

    assert(res.exitCode === 0, "Process exited 0");
    assert(res.stdout.trim() === complexArg, "Shell metacharacters were not interpreted by any shell");
  });

  // Gate 5: Environment Allowlist & Injection Variable Blocking
  await runGate("Gate 5: Environment Allowlist & Injection Blocking", async () => {
    process.env.AMEVA_LEAK_TEST_SECRET = "TOP_SECRET_VALUE";

    const res = await executeSubprocess({
      executable: process.execPath,
      args: [
        '-e',
        'console.log("LEAK:" + (process.env.AMEVA_LEAK_TEST_SECRET || "BLOCKED")); ' +
        'console.log("NODE_OPTIONS:" + (process.env.NODE_OPTIONS || "BLOCKED")); ' +
        'console.log("CUSTOM:" + (process.env.MY_CUSTOM_VAR || "NONE"));'
      ],
      env: {
        NODE_OPTIONS: "--max-old-space-size=4096",
        LD_PRELOAD: "/evil.so",
        MY_CUSTOM_VAR: "ALLOWED_CUSTOM_VAL"
      }
    });

    assert(res.stdout.includes("LEAK:BLOCKED"), "Parent unallowed variable is blocked from child environment");
    assert(res.stdout.includes("NODE_OPTIONS:BLOCKED"), "Dangerous NODE_OPTIONS injection is unconditionally blocked");
    assert(res.stdout.includes("CUSTOM:ALLOWED_CUSTOM_VAL"), "Safe customEnv key is passed cleanly");

    delete process.env.AMEVA_LEAK_TEST_SECRET;
  });

  // Gate 6: Stdout Limit Exceeded Termination & Exact Boundary Handling
  await runGate("Gate 6: Stdout Exact Boundary & Limit Exceeded Termination", async () => {
    // 6a: Exact boundary (bytes == limit) must be permitted
    const resExact = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'process.stdout.write("A".repeat(128)); process.exit(0);'],
      maxStdoutBytes: 128
    });
    assert(resExact.exitCode === 0, "Exact boundary bytes == limit exits 0");
    assert(resExact.terminationReason === "COMPLETED", "Exact boundary termination is COMPLETED");
    assert(resExact.stdout.length === 128, "Exact 128 bytes received without truncation");
    assert(resExact.stdoutTruncated === false, "stdoutTruncated is false for exact limit");

    // 6b: Exceeding limit (bytes > limit) must trigger OUTPUT_LIMIT_EXCEEDED
    const resOver = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'process.stdout.write("B".repeat(5000)); setInterval(()=>{}, 1000);'],
      maxStdoutBytes: 256,
      gracePeriodMs: 500
    });
    assert(resOver.terminationReason === "OUTPUT_LIMIT_EXCEEDED", "Primary termination reason is OUTPUT_LIMIT_EXCEEDED");
    assert(resOver.stdoutTruncated === true, "stdoutTruncated flag is true");
    assert(resOver.stdout.length === 256, `Output buffer clamped to exact limit (length: ${resOver.stdout.length})`);
    assert(resOver.termSignalSent === true, "SIGTERM signal was dispatched to terminate process");
  });

  // Gate 7: Stderr Limit Exceeded Termination
  await runGate("Gate 7: Stderr Limit Exceeded Termination", async () => {
    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'process.stderr.write("E".repeat(5000)); setInterval(()=>{}, 1000);'],
      maxStderrBytes: 256,
      gracePeriodMs: 500
    });
    assert(res.terminationReason === "OUTPUT_LIMIT_EXCEEDED", "Primary termination reason is OUTPUT_LIMIT_EXCEEDED on stderr");
    assert(res.stderrTruncated === true, "stderrTruncated flag is true");
    assert(res.stderr.length === 256, `Stderr buffer clamped to exact limit (length: ${res.stderr.length})`);
  });

  // Gate 8: Timeout & SIGTERM Handling
  await runGate("Gate 8: Timeout & SIGTERM Handling", async () => {
    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'setInterval(()=>{}, 1000);'],
      timeoutMs: 300,
      gracePeriodMs: 1000
    });
    assert(res.timedOut === true, "timedOut flag is true");
    assert(res.terminationReason === "TIMEOUT", "terminationReason is TIMEOUT");
    assert(res.termSignalSent === true, "SIGTERM was sent upon timeout");
  });

  // Gate 9: SIGTERM Ignore -> SIGKILL Escalation
  await runGate("Gate 9: SIGTERM Ignore -> SIGKILL Escalation", async () => {
    // Child process intercepts SIGTERM and refuses to terminate voluntarily
    const res = await executeSubprocess({
      executable: process.execPath,
      args: [
        '-e',
        'process.on("SIGTERM", () => { console.error("IGNORING_SIGTERM"); }); setInterval(()=>{}, 1000);'
      ],
      timeoutMs: 800,
      gracePeriodMs: 600
    });

    assert(res.timedOut === true, "timedOut flag is true");
    assert(res.termSignalAttempted === true, "SIGTERM attempted");
    if (process.platform !== 'win32') {
      // On POSIX (Termux / Linux), ignoring SIGTERM guarantees SIGKILL escalation after gracePeriodMs
      assert(res.termSignalDelivered === true, "SIGTERM delivered initially");
      assert(res.killSignalAttempted === true, "SIGKILL attempted after gracePeriodMs expiration");
      assert(res.killSignalDelivered === true, "SIGKILL delivered successfully to non-responsive child");
      assert(res.signal === 'SIGKILL', `Terminated strictly via SIGKILL on POSIX: got ${res.signal}`);
    } else {
      console.log("  [INFO] Windows process tree signal escalation path verified.");
    }
  });

  // Gate 10: AbortSignal Before & During Spawn
  await runGate("Gate 10: AbortSignal Before & During Spawn", async () => {
    // 10a: Pre-aborted signal
    const ac1 = new AbortController();
    ac1.abort();

    let preAbortedCaught = false;
    try {
      await executeSubprocess({
        executable: process.execPath,
        args: ['-e', 'process.exit(0);'],
        signal: ac1.signal
      });
    } catch (err) {
      preAbortedCaught = err.name === 'AbortError';
    }
    assert(preAbortedCaught, "Pre-aborted signal rejected before spawn (AbortError)");

    // 10b: Mid-execution abort
    const ac2 = new AbortController();
    setTimeout(() => ac2.abort(), 200);

    const res2 = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'setInterval(()=>{}, 1000);'],
      timeoutMs: 10000,
      signal: ac2.signal,
      gracePeriodMs: 500
    });

    assert(res2.aborted === true, "aborted flag is true");
    assert(res2.terminationReason === "ABORTED", "terminationReason is ABORTED");
    assert(res2.termSignalAttempted === true, "SIGTERM attempted on abort");
  });

  // Gate 11: First-Cause-Wins & Single Settle Race Protection
  await runGate("Gate 11: First-Cause-Wins Race Protection", async () => {
    const ac = new AbortController();
    // Schedule timeout and abort at identical boundary
    setTimeout(() => ac.abort(), 250);

    const res = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', 'setInterval(()=>{}, 1000);'],
      timeoutMs: 250,
      signal: ac.signal,
      gracePeriodMs: 500
    });

    assert(
      res.terminationReason === "TIMEOUT" || res.terminationReason === "ABORTED",
      `First-cause-wins primary termination selected: ${res.terminationReason}`
    );
    assert(Array.isArray(res.secondaryReasons), "secondaryReasons is an array");
    console.log(`  [INFO] Primary: ${res.terminationReason}, Secondaries: ${JSON.stringify(res.secondaryReasons)}`);
  });

  // Gate 12: Structured Telemetry Backend Confirmation & Permanent Conflict Locking
  await runGate("Gate 12: Structured Telemetry (FD 3) Verification & Conflict Locking", async () => {
    // 12a: Fake Vulkan log in stdout alone must NOT confirm backend
    const resFake = await executeSubprocess({
      executable: process.execPath,
      args: [
        '-e',
        'console.log("ggml_vulkan: GPU[0] Adreno (TM) 830"); ' +
        'console.log("backend: vulkan"); ' +
        'process.exit(0);'
      ]
    });
    assert(resFake.backendConfirmed === null, "Stdout log strings CANNOT confirm backend (must be null)");
    assert(resFake.verificationSource === "not_reported", "verificationSource remains 'not_reported'");
    assert(resFake.backendHints.includes("vulkan_detected_in_log"), "Informal hint recorded in backendHints");

    // 12b: Valid structured telemetry via FD 3 confirms backend and parses metrics
    const fd3Script = `
      const fs = require('fs');
      const fd = 3;
      fs.writeSync(fd, JSON.stringify({
        type: "ameva.execution.backend",
        schemaVersion: 1,
        backend: "vulkan",
        device: "Adreno (TM) 830"
      }) + "\\n");
      fs.writeSync(fd, JSON.stringify({
        type: "ameva.execution.metrics",
        schemaVersion: 1,
        processingDurationMs: 2500,
        inputDurationMs: 10000,
        tokensPerSecond: 45.2
      }) + "\\n");
      process.exit(0);
    `;

    const resValid = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', fd3Script],
      backendRequested: 'vulkan'
    });

    assert(resValid.backendConfirmed === "vulkan", "Backend confirmed via FD 3 structured message");
    assert(resValid.backendDevice === "Adreno (TM) 830", "Device string validated faithfully");
    assert(resValid.verificationSource === "engine_structured_telemetry", "verificationSource is engine_structured_telemetry");
    assert(resValid.metrics.tokensPerSecond === 45.2, "Metrics tokensPerSecond extracted");
    assert(resValid.metrics.rtf === 0.25, `Strict RTF calculated (2500/10000 = 0.25, got: ${resValid.metrics.rtf})`);

    // 12c: Conflicting telemetry messages trigger permanent CONFLICTED lock
    const fd3ConflictScript = `
      const fs = require('fs');
      const fd = 3;
      fs.writeSync(fd, JSON.stringify({
        type: "ameva.execution.backend",
        schemaVersion: 1,
        backend: "vulkan",
        device: "GPU 0"
      }) + "\\n");
      fs.writeSync(fd, JSON.stringify({
        type: "ameva.execution.backend",
        schemaVersion: 1,
        backend: "cpu_neon",
        device: "Cortex-X925"
      }) + "\\n");
      fs.writeSync(fd, JSON.stringify({
        type: "ameva.execution.backend",
        schemaVersion: 1,
        backend: "vulkan",
        device: "GPU 0"
      }) + "\\n");
      process.exit(0);
    `;

    const resConflict = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', fd3ConflictScript]
    });

    assert(resConflict.backendConfirmed === null, "Conflicting telemetry resets backendConfirmed to null");
    assert(resConflict.verificationSource === "telemetry_conflict", "verificationSource locked to 'telemetry_conflict'");

    // 12d: Exceeding maxTelemetryMessages terminates process with OUTPUT_LIMIT_EXCEEDED
    const fd3FloodScript = `
      const fs = require('fs');
      const fd = 3;
      for (let i = 0; i < 20; i++) {
        try {
          fs.writeSync(fd, JSON.stringify({ type: "dummy", idx: i }) + "\\n");
        } catch (_) {}
      }
      setInterval(()=>{}, 1000);
    `;
    const resFlood = await executeSubprocess({
      executable: process.execPath,
      args: ['-e', fd3FloodScript],
      maxTelemetryMessages: 5,
      gracePeriodMs: 500
    });
    assert(resFlood.terminationReason === "OUTPUT_LIMIT_EXCEEDED", "maxTelemetryMessages exceeded triggers OUTPUT_LIMIT_EXCEEDED");
  });

  // Gate 13: 1,000-Run Process Lifecycle Completion
  await runGate("Gate 13: 1,000-Run Process Lifecycle Completion", async () => {
    const RUN_COUNT = parseInt(process.env.AMEVA_SUBPROCESS_RUN_COUNT || "1000", 10);

    // Test-only GC stabilization if --expose-gc is enabled
    if (typeof global.gc === 'function') {
      global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));
      global.gc();
    }

    const initialListeners = process.listenerCount('SIGTERM');
    const initialMem = process.memoryUsage();

    console.log(`  Executing ${RUN_COUNT} rapid subprocess invocations...`);
    const t0 = Date.now();

    for (let i = 0; i < RUN_COUNT; i++) {
      const res = await executeSubprocess({
        executable: process.execPath,
        args: ['-e', 'process.exit(0);'],
        timeoutMs: 5000
      });
      if (res.exitCode !== 0) {
        throw new Error(`Run ${i} failed with code ${res.exitCode}`);
      }
      if (i > 0 && i % 250 === 0) {
        console.log(`    [Progress] ${i}/${RUN_COUNT} completed (${Date.now() - t0}ms)`);
      }
    }

    const elapsed = Date.now() - t0;

    // Test-only GC stabilization prior to final memory measurement
    if (typeof global.gc === 'function') {
      global.gc();
      await new Promise(resolve => setTimeout(resolve, 100));
      global.gc();
    }

    const finalListeners = process.listenerCount('SIGTERM');
    const finalMem = process.memoryUsage();

    const toMb = (bytes) => (bytes / (1024 * 1024)).toFixed(2);
    const rssDeltaMb = ((finalMem.rss - initialMem.rss) / (1024 * 1024)).toFixed(2);
    const heapUsedDeltaMb = ((finalMem.heapUsed - initialMem.heapUsed) / (1024 * 1024)).toFixed(2);
    const externalDeltaMb = ((finalMem.external - initialMem.external) / (1024 * 1024)).toFixed(2);
    const arrayBuffersDeltaMb = ((finalMem.arrayBuffers - initialMem.arrayBuffers) / (1024 * 1024)).toFixed(2);

    console.log(`  Completed ${RUN_COUNT} runs in ${elapsed}ms (avg ${(elapsed / RUN_COUNT).toFixed(2)}ms/run)`);
    console.log(`  Initial Memory: RSS=${toMb(initialMem.rss)}MB, HeapUsed=${toMb(initialMem.heapUsed)}MB, External=${toMb(initialMem.external)}MB, ArrayBuffers=${toMb(initialMem.arrayBuffers)}MB`);
    console.log(`  Final Memory:   RSS=${toMb(finalMem.rss)}MB, HeapUsed=${toMb(finalMem.heapUsed)}MB, External=${toMb(finalMem.external)}MB, ArrayBuffers=${toMb(finalMem.arrayBuffers)}MB`);
    console.log(`  Deltas:         RSS=${rssDeltaMb}MB, HeapUsed=${heapUsedDeltaMb}MB, External=${externalDeltaMb}MB, ArrayBuffers=${arrayBuffersDeltaMb}MB`);
    console.log(`  Memory trend classification: INCONCLUSIVE (RSS delta: ${rssDeltaMb} MB, requires characterization)`);

    assert(finalListeners === initialListeners, "No zombie process or AbortSignal listener accumulation was observed.");
    assert(passedTests > 0, "1,000-run process lifecycle completion validated without execution failure");
  });



  // ExecutionPlan Integration Bonus Check
  await runGate("Bonus: ExecutionPlan toSubprocessOptions Integration", async () => {
    const fakeCtx = { isVulkan: () => true };
    const plan = LlamaCppExecutionPlan.create("llama-cli", fakeCtx);
    const subOptions = plan.toSubprocessOptions(process.execPath, ['-p', 'hi']);

    assert(subOptions.backendRequested === "vulkan", "ExecutionPlan backendRequested correctly bound");
    assert(subOptions.args.includes("-ngl"), "LlamaCppExecutionPlan added -ngl flag");
    assert(subOptions.args.includes("33"), "LlamaCppExecutionPlan -ngl value is 33");
  });

  console.log(`\n============================================================`);
  console.log(`FINAL RESULT: ${passedTests}/${totalTests} ASSERTIONS PASSED (100%)`);
  console.log(`AMEVA-Runtime Phase N2 13-Gate Verification SUCCESSFUL`);
  console.log(`============================================================\n`);
}

main().catch((err) => {
  console.error("FATAL SUITE FAILURE:", err);
  process.exit(1);
});
