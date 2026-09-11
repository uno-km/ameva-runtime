const assert = require('assert');
const { Doctor, isAvailable, createContext, PlatformNotSupportedError } = require('./index');
const nativeBridge = require('./lib/native_bridge');

async function runPublicApiTestSuite() {
  console.log('[TEST] === Phase N1.2: Node.js Public API Unification E2E Verification Suite ===\n');
  let passCount = 0;

  function recordPass(desc) {
    passCount++;
    console.log(`  [PASS ${passCount}] ${desc}`);
  }

  // 1. Doctor.runSelfTest() invokes Native Doctor when addon is loaded
  console.log('  -> Executing new Doctor().runSelfTest(false)...');
  const doc = new Doctor();

  if (nativeBridge.isNativeLoaded()) {
    const report = await doc.runSelfTest(false);
    assert(report !== null && typeof report === 'object');
    assert.strictEqual(report.verificationSource, 'native_c_hal', 'Expected native_c_hal verificationSource');
    assert.strictEqual(report.computeCertified, true, 'Expected computeCertified to be true');
    assert.strictEqual(report.modelCertified, false, 'Expected modelCertified to be false (V10/V11 deferred)');
    assert.strictEqual(report.overallSuccess, true, 'Expected overallSuccess to be true for compute scope');
    assert(report.deviceName.includes('Adreno'), `Expected Adreno GPU, got: ${report.deviceName}`);
    assert.strictEqual(report.recommendedBackend, 'vulkan', `Expected recommendedBackend 'vulkan', got: ${report.recommendedBackend}`);
    recordPass(`Doctor.runSelfTest() delegated to Native Doctor: device='${report.deviceName}', computeCertified=${report.computeCertified}`);
  } else {
    await assert.rejects(
      async () => await doc.runSelfTest(false),
      (err) => err instanceof PlatformNotSupportedError || err.name === 'PlatformNotSupportedError',
      'Expected PlatformNotSupportedError when native addon is unavailable'
    );
    recordPass('Doctor.runSelfTest() strictly rejected with PlatformNotSupportedError when native addon is unbuilt');
  }

  // 2. Doctor.quickProbe() and quickProbeDevice()
  const qp = doc.quickProbe();
  const qpDev = doc.quickProbeDevice();
  console.log(`  -> Doctor.quickProbe(): ${qp}, device: '${qpDev}'`);
  if (nativeBridge.isNativeLoaded()) {
    assert.strictEqual(qp, true, 'Doctor.quickProbe() must evaluate to true on certified hardware');
    assert(qpDev && qpDev.includes('Adreno'), `quickProbeDevice() must return Adreno GPU name, got: ${qpDev}`);
    recordPass(`Doctor.quickProbe() passed: available=${qp}, device='${qpDev}'`);
  } else {
    assert.strictEqual(qp, false, 'Doctor.quickProbe() must be false on pure JS uncertified host');
    recordPass('Doctor.quickProbe() safely rejected unverified pure JS environment');
  }

  // 3. isAvailable() top-level public export
  const avail = isAvailable();
  assert.strictEqual(typeof avail, 'boolean');
  if (nativeBridge.isNativeLoaded()) {
    assert.strictEqual(avail, true, 'isAvailable() must return true on certified hardware');
    recordPass('isAvailable() top-level function returned true');
  } else {
    assert.strictEqual(avail, false, 'isAvailable() must return false on uncertified host');
    recordPass('isAvailable() correctly returned false in pure JS host');
  }

  // 4. createContext({ device: "auto" })
  const ctxAuto = createContext({ device: 'auto' });
  if (nativeBridge.isNativeLoaded()) {
    assert.strictEqual(ctxAuto.selectedBackend, 'vulkan', `Expected auto to select 'vulkan', got: '${ctxAuto.selectedBackend}'`);
    assert.strictEqual(ctxAuto.isGpu, true, 'ctxAuto.isGpu should be true');
    assert.strictEqual(ctxAuto.isVulkan(), true, 'ctxAuto.isVulkan() should be true');
    assert.strictEqual(ctxAuto.selectionReason, 'vulkan_certified_hardware');
    assert(ctxAuto.deviceName.includes('Adreno'), `Expected Adreno deviceName, got: ${ctxAuto.deviceName}`);
    recordPass(`createContext({ device: 'auto' }) successfully promoted to Vulkan: device='${ctxAuto.deviceName}'`);
  } else {
    const expectedCpu = process.arch === 'arm64' ? 'cpu_neon' : 'cpu_reference';
    assert.strictEqual(ctxAuto.selectedBackend, expectedCpu);
    assert.strictEqual(ctxAuto.isGpu, false);
    recordPass(`createContext({ device: 'auto' }) safely routed to ${expectedCpu} on uncertified host`);
  }

  // 5. createContext({ device: "vulkan" })
  if (nativeBridge.isNativeLoaded()) {
    const ctxVk = createContext({ device: 'vulkan' });
    assert.strictEqual(ctxVk.selectedBackend, 'vulkan');
    assert.strictEqual(ctxVk.selectionReason, 'explicit_gpu_request_verified');
    recordPass("createContext({ device: 'vulkan' }) explicitly verified and selected Vulkan");
  } else {
    assert.throws(
      () => createContext({ device: 'vulkan' }),
      PlatformNotSupportedError,
      'Expected PlatformNotSupportedError on explicit vulkan in pure JS host'
    );
    recordPass("createContext({ device: 'vulkan' }) threw PlatformNotSupportedError in pure JS host (Fail-Fast)");
  }

  // 6. createContext({ device: "cpu" })
  const expectedCpu = process.arch === 'arm64' ? 'cpu_neon' : 'cpu_reference';
  const ctxCpu = createContext({ device: 'cpu' });
  assert.strictEqual(ctxCpu.selectedBackend, expectedCpu);
  assert.strictEqual(ctxCpu.isGpu, false);
  assert.strictEqual(ctxCpu.selectionReason, 'explicit_cpu_request');
  recordPass(`createContext({ device: 'cpu' }) explicitly selected ${expectedCpu}`);

  console.log(`\n[SUCCESS] All ${passCount}/6 Public API Unification criteria passed!`);
}

if (require.main === module) {
  runPublicApiTestSuite().catch(err => {
    console.error('\n[FATAL] Public API Test Suite Failed:', err);
    process.exit(1);
  });
}

module.exports = { runPublicApiTestSuite };
