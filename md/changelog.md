# Changelog

Fork release history.

## v1.0.0 — 2026-10-01
- **Summary:** Initial fork release **v1.0.0** / package **`spas_sage_hswq_attn`** v1.0.0 — Native Windows / MSVC build pipeline, complete 7-generation architecture coverage (Blackwell included), package namespace isolation, and bit-width-preserving quantization enhancements:
  - **Native Windows / MSVC Build Support:** Resolved Windows `CreateProcess` 32,767-character limit by monkey-patching `link.exe` command execution via response files (`@link.rsp`), overcoming `LNK1104` link failures across 219 kernel objects. Guarded `_GLIBCXX_USE_CXX11_ABI` on Windows, resolving CCCL `cuda/std/__fwd/string.h` macro collision syntax errors with MSVC STL on CUDA 13.x.
  - **Full 7-Generation GPU Architecture Coverage:** 100% full-spec fatbin inclusion covering Ampere (`sm_80`, `sm_86`), Ada Lovelace (`sm_89`), Hopper (`sm_90a`), and Blackwell (`sm_100`, `sm_120`, `sm_121`) under `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;10.0;12.0;12.1"` (~34MB complete wheel footprint).
  - **Package Namespace Isolation:** Renamed package and extension modules from `spas_sage_attn` to `spas_sage_hswq_attn` (`_qattn`, `_fused`) to prevent binary collisions when installed alongside upstream official packages or standalone `sageattention` in shared environments.
  - **Bit-Width-Preserving Accuracy Enhancements:** Integrated per-block INT8 scale sweep (`scale_sweep.py`, E2-a) minimizing reconstruction MSE against FP16 ground truth, threshold headroom calibration (`headroom.py`, E2-b) eliminating visual flicker artifacts from edge-case activation spikes, and layer-importance weighting (E2-d).
  - **Dynamic Flash-Attention Style Versioning & Pre-built Wheels:** Embedded dynamic metadata resolution generating wheel tags compliant with Flash-Attention standards (`+cu<CUDA>torch<TORCH>cxx11abi<ABI>`). Pre-built wheels provided for Python 3.13 and Python 3.14.
- **Technical Details:** See [v1.0.0 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0.0) for complete explanation
