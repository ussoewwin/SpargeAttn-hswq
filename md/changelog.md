# Changelog

All notable changes to the `SpargeAttn-hswq` project will be documented in this file.

---

## [v1.0.0] - 2026-10-01

- **Release Notes**: [v1.0.0 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0.0)

### Key Improvements over Official SpargeAttn

- **Native Windows / MSVC Build**: Bypassed MSVC 32k linker command-line limit via response files (`@link.rsp`) and resolved CUDA 13.x CCCL Windows ABI issues.
- **Full 7-Generation Architecture Coverage**: 100% full-spec fatbin inclusion covering Ampere, Ada Lovelace, Hopper, and Blackwell (`sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100`, `sm_120`, `sm_121`).
- **Namespace Isolation**: Renamed to `spas_sage_hswq_attn` to safely coexist with official upstream packages.
- **Bit-Width-Preserving Enhancements**: Added INT8 block scale-sweep (E2-a), threshold headroom calibration (E2-b), and layer-importance weighting (E2-d).
- **Pre-built Wheels**: Provided pre-built Windows wheels for Python 3.13 and Python 3.14 with Flash-Attention standard naming (`+cu132torch2.14.0...`).
