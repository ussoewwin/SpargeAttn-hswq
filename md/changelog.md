# Changelog

Fork release history.

## v1.0.0 — 2026-10-01
- **Summary:** Initial fork release **v1.0.0** / package **`spas_sage_hswq_attn`** v1.0.0:
  - **Native Windows / MSVC Build Support:** Resolved Windows 32k command-line limit via response file substitution (`@link.rsp`); guarded `_GLIBCXX_USE_CXX11_ABI` for CUDA 13.x compatibility.
  - **Full 7-Generation Architecture Coverage:** Native fatbin support for Ampere, Ada, Hopper, and Blackwell (`sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100`, `sm_120`, `sm_121`).
  - **Package Namespace Isolation:** Renamed package and extensions to `spas_sage_hswq_attn` to prevent collisions with upstream packages.
  - **Bit-Width-Preserving Enhancements:** INT8 scale sweep, threshold headroom calibration, and layer-importance weighting.
  - **Pre-built Windows Wheels:** Python 3.13 & 3.14 wheels with Flash-Attention standard naming convention.
- **Technical Details:** See [v1.0.0 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0.0) for complete explanation
