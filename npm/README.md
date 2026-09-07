# AMEVA-Runtime (Node.js & TypeScript)

[![npm](https://img.shields.io/npm/v/@ameva/runtime.svg?style=flat-square&color=b91c1c)](https://www.npmjs.com/package/@ameva/runtime)
[![License](https://img.shields.io/badge/License-Apache_2.0-004499.svg?style=flat-square)](https://github.com/uno-km/ameva-runtime)

> Next-Gen Unified On-Device Hardware Orchestration & 6-Modality AI Acceleration Runtime for Mobile & Edge

## Installation

```bash
npm install @ameva/runtime
```

## Quickstart

```typescript
import { Doctor, createContext } from '@ameva/runtime';

const doc = new Doctor();
const report = await doc.runSelfTest();
console.log(`Vulkan GPU: ${report.deviceName}`);
```

## Documentation
- [Official Documentation](https://uno-km.vercel.app/lib/vulkan/)
- [GitHub Repository](https://github.com/uno-km/ameva-runtime)

## License
Apache-2.0 License. Copyright (c) 2026 Eunho Kim (@uno-km).
