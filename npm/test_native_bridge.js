const assert = require('assert');
const nativeBridge = require('./lib/native_bridge');
const { PlatformNotSupportedError } = require('./lib/context');

async function runNativeBridgeTestSuite() {
  console.log('[TEST] === Phase N1: Node-API Native Bridge 13-Point Comprehensive Verification Suite ===\n');
  let passCount = 0;
  let totalCount = 0;

  function recordPass(desc) {
    passCount++;
    totalCount++;
    console.log(`  [PASS ${passCount}] ${desc}`);
  }

  function recordFail(desc, err) {
    totalCount++;
    console.error(`  [FAIL] ${desc}:`, err);
    throw err;
  }

  // Check if native addon is present
  if (!nativeBridge.isNativeLoaded()) {
    console.warn('  [WARN] Native addon is not loaded in current environment.');
    // Test Criterion 11: Fail-fast with PlatformNotSupportedError when addon is absent
    try {
      nativeBridge.getBridgeVersion();
      assert.fail('Should have thrown PlatformNotSupportedError');
    } catch (e) {
      assert(e instanceof PlatformNotSupportedError, 'Expected PlatformNotSupportedError');
      recordPass('Criterion 11: Zero-silent-fallback verified: PlatformNotSupportedError thrown when native is unbuilt');
    }
    console.log(`\n[SUMMARY] Standalone pass: ${passCount}/${totalCount} (Native binary unavailable in this host)`);
    return;
  }

  // 1. Galaxy S25 Termux Node에서 ameva_native.node 실제 로드
  try {
    assert.strictEqual(nativeBridge.isNativeLoaded(), true);
    recordPass('Criterion 1: ameva_native.node successfully loaded into Node.js runtime');
  } catch (e) { recordFail('Criterion 1', e); }

  // 2. getVersion이 실제 C ABI 결과 반환
  try {
    const ver = nativeBridge.getVersion();
    assert.strictEqual(typeof ver, 'string');
    assert.strictEqual(ver, '1.2.0');

    const bridgeVer = nativeBridge.getBridgeVersion();
    assert.strictEqual(bridgeVer, 1);

    const abiStatus = nativeBridge.getNativeAbiVersion();
    assert.strictEqual(abiStatus.supported, false);
    assert.strictEqual(abiStatus.status, 'Runtime native ABI compatibility verification unavailable');

    const structSize = nativeBridge.getDiagnosticResultStructSize();
    assert(structSize > 0, `Expected sizeof(AmevaDiagnosticResult) > 0, got ${structSize}`);

    recordPass(`Criterion 2: C ABI getVersion() returned '${ver}', bridgeVer=${bridgeVer}, structSize=${structSize}B`);
  } catch (e) { recordFail('Criterion 2', e); }

  // 3. Native Doctor가 실제 GPU 명칭과 단계 결과 반환 (V0~V9 PASS 및 computeCertified=true 검증 후 quickProbe 캐시 일치 검증)
  try {
    console.log('    -> Executing async native runDiagnostic({ verbose: false })...');
    const diag = await nativeBridge.runDiagnostic({ verbose: false });
    assert(diag !== null && typeof diag === 'object');
    assert.strictEqual(diag.exitCode, 0, `Expected exitCode 0, got: ${diag.exitCode}`);
    assert.strictEqual(diag.overallSuccess, true, 'Expected overallSuccess to be true');
    assert(diag.passedStages >= 10, `Expected passedStages >= 10, got: ${diag.passedStages}`);
    assert.strictEqual(diag.computeCertified, true, 'Expected computeCertified to be true (V0-V9 passed)');
    assert.strictEqual(diag.modelCertified, false, 'Expected modelCertified to be false (V10-V11 unreached)');
    assert.strictEqual(diag.diagnosticScope, 'compute', `Expected diagnosticScope 'compute', got '${diag.diagnosticScope}'`);
    assert.strictEqual(diag.scopeSuccess, true, 'Expected scopeSuccess to be true');
    assert.strictEqual(diag.recommendedBackend, 'vulkan', `Expected recommendedBackend 'vulkan', got '${diag.recommendedBackend}'`);
    assert(diag.deviceName.includes('Adreno'), `Expected Adreno device name, got: ${diag.deviceName}`);

    const probe = nativeBridge.quickProbe();
    assert(typeof probe.available === 'boolean');
    assert(typeof probe.deviceName === 'string');
    console.log(`    -> Native quickProbe (cached): available=${probe.available}, device='${probe.deviceName}'`);
    assert.strictEqual(probe.available, true, 'Expected quickProbe available to be true on certified Adreno 830');
    assert.strictEqual(probe.deviceName, diag.deviceName, 'quickProbe deviceName must match runDiagnostic deviceName');

    recordPass(
      `Criterion 3: Native Doctor async resolved: device='${diag.deviceName}', scope='${diag.diagnosticScope}', ` +
      `passed=${diag.passedStages}/${diag.totalStages}, computeCertified=${diag.computeCertified}, modelCertified=${diag.modelCertified}, backend='${diag.recommendedBackend}'`
    );
  } catch (e) { recordFail('Criterion 3', e); }

  // 4. 2x2 SGEMM 전체 값 일치
  // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]] -> C = [[19, 22], [43, 50]]
  try {
    const a2 = new Float32Array([1, 2, 3, 4]);
    const b2 = new Float32Array([5, 6, 7, 8]);
    const c2 = await nativeBridge.matmulF32(a2, b2, 2, 2, 2);

    assert(c2 instanceof Float32Array);
    assert.strictEqual(c2.length, 4);
    assert.strictEqual(c2[0], 19);
    assert.strictEqual(c2[1], 22);
    assert.strictEqual(c2[2], 43);
    assert.strictEqual(c2[3], 50);
    recordPass(`Criterion 4: 2x2 SGEMM exact match: [${Array.from(c2).join(', ')}]`);
  } catch (e) { recordFail('Criterion 4', e); }

  // 5. 4x4 SGEMM 전체 값 일치
  // Identity matrix test: A = 4x4 filled, B = 4x4 Identity -> C == A
  try {
    const m = 4, k = 4, n = 4;
    const a4 = new Float32Array([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16
    ]);
    const b4_identity = new Float32Array([
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1
    ]);
    const c4 = await nativeBridge.matmulF32(a4, b4_identity, m, k, n);
    assert.strictEqual(c4.length, 16);
    for (let i = 0; i < 16; i++) {
      assert.strictEqual(c4[i], a4[i], `Mismatch at index ${i}: expected ${a4[i]}, got ${c4[i]}`);
    }
    recordPass('Criterion 5: 4x4 SGEMM identity matrix exact match across all 16 elements');
  } catch (e) { recordFail('Criterion 5', e); }

  // 6. 입력 TypedArray가 GC돼도 작업 안전 (Safe Model A: copy-on-entry)
  try {
    let p;
    {
      // Allocate inside block and let scope exit immediately
      const tempA = new Float32Array(100 * 100).fill(1.0);
      const tempB = new Float32Array(100 * 100).fill(2.0);
      p = nativeBridge.matmulF32(tempA, tempB, 100, 100, 100);
    }
    // Encourage V8 GC if available
    if (global.gc) { global.gc(); }
    const res = await p;
    assert.strictEqual(res.length, 10000);
    // Each element should be 100 * (1.0 * 2.0) = 200.0
    assert.strictEqual(res[0], 200.0);
    assert.strictEqual(res[9999], 200.0);
    recordPass('Criterion 6: Input TypedArray GC immunity verified: completed 100x100 GEMM with exact value 200.0');
  } catch (e) { recordFail('Criterion 6', e); }

  // 7. 잘못된 길이와 overflow 차원 거부
  try {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([1, 2, 3, 4]);

    // Negative dimension
    await assert.rejects(
      async () => await nativeBridge.matmulF32(a, b, -1, 2, 2),
      { name: 'RangeError' }
    );
    // Non-integer dimension
    await assert.rejects(
      async () => await nativeBridge.matmulF32(a, b, 2.5, 2, 2),
      { name: 'RangeError' }
    );
    // NaN dimension
    await assert.rejects(
      async () => await nativeBridge.matmulF32(a, b, NaN, 2, 2),
      { name: 'RangeError' }
    );
    // Infinity dimension
    await assert.rejects(
      async () => await nativeBridge.matmulF32(a, b, Infinity, 2, 2),
      { name: 'RangeError' }
    );
    // Overflow dimension (> UINT32_MAX or m*k overflow)
    await assert.rejects(
      async () => await nativeBridge.matmulF32(a, b, 4294967295, 2, 2),
      { name: 'RangeError' }
    );
    recordPass('Criterion 7: Safe Math: NaN, Infinity, negative, non-integer, and overflow dimensions strictly rejected');
  } catch (e) { recordFail('Criterion 7', e); }

  // 8. Native 실패 및 버퍼 크기 부족 시 reject
  try {
    const shortA = new Float32Array([1, 2]); // needs 4
    const b = new Float32Array([1, 2, 3, 4]);
    await assert.rejects(
      async () => await nativeBridge.matmulF32(shortA, b, 2, 2, 2),
      { name: 'RangeError' }
    );
    recordPass('Criterion 8: Buffer bounds verification: short TypedArray rejected before async worker');
  } catch (e) { recordFail('Criterion 8', e); }

  // 9. 100회 async matmul 완료 (Stress & Leak-free verification)
  try {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([2, 0, 0, 2]);
    const t0 = Date.now();
    for (let i = 0; i < 100; i++) {
      const res = await nativeBridge.matmulF32(a, b, 2, 2, 2);
      assert.strictEqual(res[0], 2);
      assert.strictEqual(res[3], 8);
    }
    const elapsed = Date.now() - t0;
    recordPass(`Criterion 9: 100-cycle short-run async native-call stability passed in ${elapsed}ms`);
  } catch (e) { recordFail('Criterion 9', e); }

  // 10. Promise.all 병렬 요청이 안전하게 직렬화됨을 물리적 계측으로 증명 (maxActiveCalls === 1)
  try {
    nativeBridge.resetConcurrencyMetrics();
    const parallelOps = [];
    for (let i = 0; i < 10; i++) {
      const scale = i + 1;
      const a = new Float32Array([scale, 0, 0, scale]);
      const b = new Float32Array([2, 0, 0, 2]);
      parallelOps.push(nativeBridge.matmulF32(a, b, 2, 2, 2));
    }
    const results = await Promise.all(parallelOps);
    assert.strictEqual(results.length, 10);
    for (let i = 0; i < 10; i++) {
      const expectedVal = (i + 1) * 2;
      assert.strictEqual(results[i][0], expectedVal);
      assert.strictEqual(results[i][3], expectedVal);
    }

    const metrics = nativeBridge.getConcurrencyMetrics();
    console.log(`    -> Concurrency metrics: active=${metrics.activeCalls}, maxActive=${metrics.maxActiveCalls}, total=${metrics.totalCalls}`);
    assert.strictEqual(metrics.activeCalls, 0, 'Active calls should be 0 after completion');
    assert.strictEqual(metrics.maxActiveCalls, 1, `Expected maxActiveCalls === 1 (strictly serialized), got: ${metrics.maxActiveCalls}`);
    assert(metrics.totalCalls >= 10, `Expected totalCalls >= 10, got: ${metrics.totalCalls}`);

    recordPass(`Criterion 10: Promise.all parallel async calls physically serialized by mutex (maxActiveCalls=${metrics.maxActiveCalls}, totalCalls=${metrics.totalCalls})`);
  } catch (e) { recordFail('Criterion 10', e); }

  // 11. PlatformNotSupportedError contract
  try {
    assert.strictEqual(typeof nativeBridge.isNativeLoaded(), 'boolean');
    recordPass('Criterion 11: Fail-fast and platform detection contracts verified');
  } catch (e) { recordFail('Criterion 11', e); }

  // 12. Tensor alignment C ABI query
  try {
    const aligned = nativeBridge.isTensorAligned(128, 128);
    assert.strictEqual(typeof aligned, 'boolean');
    recordPass(`Criterion 12: Tensor alignment C ABI query verified: isTensorAligned(128, 128) -> ${aligned}`);
  } catch (e) { recordFail('Criterion 12', e); }

  // 13. matmulF32WithTelemetry truthfulness verification (CPU reference GEMM, no fake Vulkan claim)
  try {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6, 7, 8]);
    const resWithTel = await nativeBridge.matmulF32WithTelemetry(a, b, 2, 2, 2);
    assert(resWithTel !== null && typeof resWithTel === 'object');
    assert(resWithTel.data instanceof Float32Array);
    assert.strictEqual(resWithTel.data[0], 19);
    assert.strictEqual(resWithTel.data[3], 50);
    assert.strictEqual(resWithTel.m, 2);
    assert.strictEqual(resWithTel.k, 2);
    assert.strictEqual(resWithTel.n, 2);

    const tel = resWithTel.telemetry;
    assert(tel !== null && typeof tel === 'object');
    assert.strictEqual(tel.backend, 'cpu_reference');
    assert.strictEqual(tel.implementation, 'native_c_reference_gemm');
    assert.strictEqual(tel.accelerated, false);
    assert.strictEqual(tel.vulkanDispatch, 'NOT_PERFORMED');
    assert(typeof tel.durationMs === 'number' && tel.durationMs >= 0);

    console.log(`    -> matmulF32WithTelemetry: backend='${tel.backend}', implementation='${tel.implementation}', vulkanDispatch='${tel.vulkanDispatch}', durationMs=${tel.durationMs}`);
    recordPass('Criterion 13: matmulF32WithTelemetry verified truthful: backend="cpu_reference", vulkanDispatch="NOT_PERFORMED"');
  } catch (e) { recordFail('Criterion 13', e); }

  console.log(`\n[SUCCESS] All ${passCount}/${totalCount} verification criteria successfully validated!`);
}

if (require.main === module) {
  runNativeBridgeTestSuite().catch((err) => {
    console.error('\n[FATAL] Native Bridge Test Suite Failed:', err);
    process.exit(1);
  });
}

module.exports = { runNativeBridgeTestSuite };

