# Changelog

## v1.0.0 — 2026-10-01

- **Summary:** Initial fork release **v1.0.0** / package **`spas_sage_hswq_attn` 1.0.0** — Native Windows / MSVC build support, full 7-generation architecture coverage (Blackwell included), namespace isolation, and bit-width-preserving quantization enhancements.
  - **Native Windows / MSVC build pipeline:** Overcame the Windows 32k command-line limit via automatic response file substitution (`@link.rsp`) in MSVC linker invocation, resolving `LNK1104` link failures across 219 kernel objects. Resolved CUDA 13.x CCCL Windows ABI conflict (`_GLIBCXX_USE_CXX11_ABI` guard).
  - **Full 7-generation architecture support:** Native code generation for Ampere (`sm_80`, `sm_86`), Ada Lovelace (`sm_89`), Hopper (`sm_90a`), and Blackwell (`sm_100`, `sm_120`, `sm_121`) with 100% full-spec fatbin inclusion (~34MB complete wheel footprint).
  - **Package namespace isolation:** Renamed package and extensions to `spas_sage_hswq_attn` (`_qattn`, `_fused`) to prevent collisions with upstream official packages in shared environments.
  - **Bit-width-preserving enhancements:** Integrated INT8 per-block scale sweep (`scale_sweep.py`), threshold headroom calibration (`headroom.py`), and layer-importance weighting.
  - **Pre-built Windows wheels:** Generated wheels for Python 3.13 and Python 3.14 with Flash-Attention standard naming (`+cu132torch2.14.0...`).
- **Release (GitHub):** https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0.0
