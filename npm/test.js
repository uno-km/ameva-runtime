const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');
const {
  Doctor,
  createContext,
  getOrCreateContext,
  isAvailable,
  SttExecutionPlan,
  DiffusionExecutionPlan,
  BitnetExecutionPlan,
  LlamaCppExecutionPlan,
  TtsExecutionPlan,
  VisionExecutionPlan,
  SttAdapter,
  DiffusionAdapter,
  BitnetAdapter,
  LlamaCppAdapter,
  TtsAdapter,
  VisionAdapter,
  nativeBridge
} = require('./index');

async function runTests() {
  console.log("[TEST] Running AMEVA Vulkan Runtime Node.js Test Suite (Rigorous Honesty Verification)...");

  const isNative = nativeBridge.isNativeLoaded();

  // 1. Doctor Test
  const doc = new Doctor();
  const report = await doc.runSelfTest(false);
  assert.strictEqual(typeof report.overallSuccess, 'boolean', "overallSuccess must be boolean");
  if (isNative) {
    assert.strictEqual(report.overallSuccess, true, "Native Doctor must report overallSuccess=true on compute-certified GPU");
    assert.strictEqual(report.computeCertified, true, "Native Doctor must certify compute");
    assert.strictEqual(doc.quickProbe(), true, "Doctor.quickProbe() must be true on certified hardware");
  } else {
    assert.strictEqual(report.overallSuccess, false, "Pure JS runSelfTest must honestly report overallSuccess=false");
    assert.strictEqual(doc.quickProbe(), false, "Pure JS probe without native_c_hal must return false in quickProbe()");
  }
  assert.strictEqual(typeof report.passedStages, 'number', "passedStages must be number");
  assert.strictEqual(Array.isArray(report.stages), true, "stages must be an array");

  // 2. CPU Mode Bypass Test
  const cpuCtx = createContext({ device: "cpu" });
  assert.strictEqual(cpuCtx.isGpu, false, "CPU mode should not be GPU");
  assert.strictEqual(cpuCtx.backendType, "cpu_neon", "CPU backend type should be cpu_neon");
  assert.strictEqual(cpuCtx.selectedBackend, "cpu_neon");
  assert.strictEqual(cpuCtx.selectionReason, "explicit_cpu_request");
  const whisperFlags = cpuCtx.toEngineFlags("whisper");
  assert.strictEqual(whisperFlags.useGpu, false, "whisper flags should reflect CPU");
  assert.strictEqual(whisperFlags.gpuLayers, 0, "whisper gpuLayers should be 0 on CPU");

  // 3. Auto Mode Honest Routing Test
  const autoCtx = createContext({ device: "auto" });
  assert.strictEqual(typeof autoCtx.isGpu, 'boolean', "isGpu must be boolean");
  if (isNative) {
    assert.strictEqual(autoCtx.selectedBackend, "vulkan", "Certified GPU must auto-route to vulkan");
    assert.strictEqual(autoCtx.selectionReason, "vulkan_certified_hardware");
  } else {
    assert.strictEqual(autoCtx.selectedBackend, "cpu_neon", "Unverified GPU environment must auto-route to cpu_neon");
    assert.strictEqual(autoCtx.selectionReason, "vulkan_probe_unverified");
  }

  // 4. getOrCreateContext reuse
  const reused = getOrCreateContext(cpuCtx);
  assert.strictEqual(reused, cpuCtx, "getOrCreateContext should reuse instance");

  // 5. Memory Budget Validation (validateBufferBudget & deprecated allocateBuffer)
  const budgetCheck = cpuCtx.validateBufferBudget(1024 * 1024 * 50); // 50MB
  assert.strictEqual(budgetCheck, 1024 * 1024 * 50);
  assert.throws(() => {
    cpuCtx.validateBufferBudget(1024 * 1024 * 2048); // 2048MB > 1024MB
  }, /exceeds configured memory limit/);

  // Deprecated allocateBuffer call test
  const legacyBuf = cpuCtx.allocateBuffer(1024 * 1024 * 10);
  assert.strictEqual(legacyBuf, 1024 * 1024 * 10);

  // 6. ExecutionPlan Binding Tests
  const stt = SttExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(stt.isVulkan, false);
  assert.strictEqual(stt.backend, "cpu_neon");
  assert.strictEqual(stt.status, "PLANNED");
  assert.strictEqual(stt.executionStatus, "NOT_EXECUTED");
  assert.strictEqual(stt.rtfTarget, undefined, "Fixed fake RTF target must be deleted");

  const diff = DiffusionExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(diff.status, "PLANNED");
  assert.strictEqual(diff.executionStatus, "NOT_EXECUTED");

  const bit = BitnetExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(bit.status, "PLANNED");

  const llama = LlamaCppExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(llama.status, "PLANNED");

  const tts = TtsExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(tts.status, "PLANNED");
  assert.strictEqual(tts.latencyMs, undefined, "Fixed fake latencyMs must be deleted");

  const vis = VisionExecutionPlan.create(null, cpuCtx);
  assert.strictEqual(vis.status, "PLANNED");

  // 7. Comprehensive Symmetric Test Suite: Positive Production Path & Negative Perturbations
  const tempStatePath = path.join(os.tmpdir(), `test_ameva_state_${Date.now()}.json`);
  const dummyDriver = path.join(os.tmpdir(), `dummy_libvulkan_${Date.now()}.so`);
  const origEnvLib = process.env.AMEVA_VULKAN_LIB;
  try {
    fs.writeFileSync(dummyDriver, "DUMMY_NATIVE_VULKAN_DRIVER_BINARY");
    const realDriverPath = fs.realpathSync(dummyDriver);
    process.env.AMEVA_VULKAN_LIB = realDriverPath;

    // Baseline: Valid Canonical Schema Version 2 Production State
    const canonicalValidReport = {
      schemaVersion: 2,
      overallSuccess: true,
      verificationSource: "native_c_hal",
      verifiedAt: new Date().toISOString(),
      deviceFingerprint: {
        vendorId: 0x5143,
        deviceId: 0x06050000,
        driverVersion: 0x80350000,
        apiVersion: 0x00403000,
        deviceName: "Qualcomm Adreno (TM) 830",
        loaderPath: realDriverPath
      },
      validation: {
        passedStages: 10,
        totalStages: 10
      },
      recommendedBackend: "vulkan"
    };

    // [Positive Test]: All criteria met -> MUST return true (proves implementation is NOT always-false dummy)
    fs.writeFileSync(tempStatePath, JSON.stringify(canonicalValidReport, null, 2), 'utf-8');
    const validDoc = new Doctor(tempStatePath);
    assert.strictEqual(validDoc.quickProbe(), true, "[POSITIVE TEST FAIL] Valid canonical state MUST yield quickProbe() === true");

    // [Negative Test 1]: schemaVersion !== 2 -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({ ...canonicalValidReport, schemaVersion: 1 }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (schemaVersion=1) must return false");

    // [Negative Test 2]: overallSuccess !== true -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({ ...canonicalValidReport, overallSuccess: false }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (overallSuccess=false) must return false");

    // [Negative Test 3]: verificationSource !== native_c_hal -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({ ...canonicalValidReport, verificationSource: "pure_js_probe" }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (verificationSource=pure_js_probe) must return false");

    // [Negative Test 4]: passedStages < totalStages -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      validation: { passedStages: 9, totalStages: 10 }
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (passedStages=9 < 10) must return false");

    // [Negative Test 5]: totalStages < 10 -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      validation: { passedStages: 6, totalStages: 6 }
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (totalStages=6 < 10) must return false");

    // [Negative Test 6]: loaderPath does not exist on host -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      deviceFingerprint: { ...canonicalValidReport.deviceFingerprint, loaderPath: "/system/lib64/nonexistent_libvulkan.so" }
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (nonexistent loaderPath) must return false");

    // [Negative Test 7]: vendorId <= 0 -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      deviceFingerprint: { ...canonicalValidReport.deviceFingerprint, vendorId: 0 }
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (vendorId=0) must return false");

    // [Negative Test 8]: deviceId is null/undefined -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      deviceFingerprint: { ...canonicalValidReport.deviceFingerprint, deviceId: null }
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (deviceId=null) must return false");

    // [Negative Test 9]: TTL expired (> 24 hours) -> MUST return false
    fs.writeFileSync(tempStatePath, JSON.stringify({
      ...canonicalValidReport,
      verifiedAt: new Date(Date.now() - 25 * 3600 * 1000).toISOString()
    }, null, 2), 'utf-8');
    assert.strictEqual(validDoc.quickProbe(), false, "Negative perturbation (TTL expired > 24h) must return false");

  } finally {
    if (origEnvLib !== undefined) process.env.AMEVA_VULKAN_LIB = origEnvLib;
    else delete process.env.AMEVA_VULKAN_LIB;
    if (fs.existsSync(tempStatePath)) fs.unlinkSync(tempStatePath);
    if (fs.existsSync(dummyDriver)) fs.unlinkSync(dummyDriver);
  }

  console.log("[PASS] All Node.js tests passed successfully (Honest ExecutionPlan, Strict C HAL quickProbe, Budget Validator, Zero Fake Metrics, Symmetric Positive/Negative Validation).");
}

runTests().catch(err => {
  console.error("[FAIL]", err);
  process.exit(1);
});
