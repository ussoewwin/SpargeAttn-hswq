# Changelog

## v1.2 — 2026-10-04

- **Summary:** Blackwell (sm100/sm120/sm121) correctness & performance release — the SageAttention2++ fp16-accumulate kernel is now actually built and used on sm120
  - **Root cause of the slowness on Blackwell:** the Sage2++ kernel (`qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold`) was guarded out at build time in the previous wheel, so `_qattn` shipped without the symbol, `SAGE2PP_ENABLED` silently resolved to `False` at import ("Warning: Sage2++ NOT enabled"), and every call fell back to the slower `f32`-accumulate kernel — which made spargeattn slower than SageAttention2 even though it is the evolved form
  - **Fix:** rebuilt the package with `-DSAGE2PP_ENABLED` for sm120 (TORCH_CUDA_ARCH_LIST=12.0, CUDA 13.2). The fp16-accumulate kernel is now present in `_qattn` and `SAGE2PP_ENABLED` resolves to `True` at import
  - **Measured on RTX 5060 Ti (sm120), torch 2.14.1+cu132, 25×4032:** with the Sage2++ kernel enabled, spargeattn beats sageattn_2 (e.g. topk=0.5: 51.6 ms vs 70.3 ms = 1.36x; topk=0.25: 42.6 ms vs 71.0 ms = 1.67x)
  - Reverted an experimental batched block-map path in the SeedVR2 integration: `spas_sage2_attn_meansim_topk_cuda` already runs one window per call and internally uses the Sage2++ fp16-accumulate kernel, so the batched re-implementation only added `torch.stack` copies (q/k/v are `(total_seq,H,D)` and NA windows are ragged) and made spargeattn slower than the stock path in practice
- **Release Notes:** [v1.2 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.2) (to be published)

## v1.1 — 2026-10-03

- **Summary:** Python-host optimization release (no kernel algorithm changes; fixed-measured faster than SageAttention2 at every sequence length)
  - Top-k block selection: replaced the per-call sort + cumsum + searchsorted chain with a single `torch.topk` + scatter (removes ~3 ms fixed Stage-1 overhead, dominant for 4k-16k token sequences on Windows/WDDM); scalar `topk` avoids the `.item()` device sync; `topk=1.0` short-circuits the selection kernels entirely
  - bf16 end-to-end: bf16 inputs no longer round-trip V through fp16 on sm89+ (fused fp8 kernels template on bf16); Ampere (sm80/86/87) keeps the required fp16 V conversion behind an arch guard
  - Per-device arch-detection cache, conditional `torch.cuda.set_device`, and scalar hyperparameter-tensor caching (removes per-call GPU enumeration and allocations)
  - `output_dtype` is now honored (previously ignored); default changed from `torch.float16` to `None` (= input dtype), fixing silent bf16→fp16 output casts
  - Verified on RTX 5060 Ti (sm_120), torch 2.14.1+cu132: faster than SageAttention2 at 4k-75k tokens (e.g. 4k: 1.58 ms vs 2.05 ms; 75k: 315 ms vs 531 ms), topk=1.0 numerically matches SA2 (L1 0.0371 / cos 0.99931), block-map agreement with the stock selection = 1.0
- **Release Notes:** [v1.1 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.1)

## v1.0 — 2026-10-01

- **Summary:** Initial fork release (`spas_sage_hswq_attn` v1.0)
  - Native Windows / MSVC build pipeline (CUDA 13.x compatibility, linker response file support)
  - Full 7-generation GPU architecture support including Blackwell (`sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100`, `sm_120`, `sm_121`)
  - Package namespace isolation (`spas_sage_hswq_attn`)
  - Bit-width-preserving quantization enhancements (per-block INT8 scale sweep, threshold headroom calibration, layer-importance weighting)
  - Pre-built Windows wheels for Python 3.13 and 3.14 with Flash-Attention standard naming
- **Release Notes:** [v1.0 Release Notes](https://github.com/ussoewwin/SpargeAttn-hswq/releases/tag/v1.0)
