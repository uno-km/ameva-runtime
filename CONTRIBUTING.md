# Contributing to AMEVA Runtime

Thank you for your interest in contributing to `@ameva/runtime`!

## Code of Conduct & Ground-Truth Engineering
1. **Zero-Hype Standard**: All PRs, documentation, and performance claims must be backed by reproducible benchmarks and hardware telemetry.
2. **Zero-Silent-Fallback**: Native failures must produce explicit errors (Fail-Fast) rather than silently falling back to mock or emulation modes.
3. **Controlled Execution**: External subprocess execution must strictly follow the Controlled Subprocess Lifecycle (`shell: false`, absolute executable paths, First-Cause-Wins termination).

## Submitting Pull Requests
1. Fork the repository and create your branch from `main`.
2. Ensure all regression tests pass (`node test.js` in `npm/`).
3. Maintain zero-warning clean compilation for native C++ bindings (`node-gyp rebuild`).
4. Sign your commits (`git commit -s`).
