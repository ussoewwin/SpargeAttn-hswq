# Changelog

## v1.0 — 2026-10-01

- **Summary:** Initial fork release (`spas_sage_hswq_attn` v1.0)
  - Native Windows / MSVC build pipeline (CUDA 13.x compatibility, linker response file support)
  - Full 7-generation GPU architecture support including Blackwell (`sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100`, `sm_120`, `sm_121`)
  - Package namespace isolation (`spas_sage_hswq_attn`)
  - Bit-width-preserving quantization enhancements (per-block INT8 scale sweep, threshold headroom calibration, layer-importance weighting)
  - Pre-built Windows wheels for Python 3.13 and 3.14 with Flash-Attention standard naming
- **Release Notes:** [v1.0 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0)
