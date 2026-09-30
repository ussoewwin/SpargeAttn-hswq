# SpargeAttn-hswq v1.0.0 — Technical Documentation: What Was Changed from Official SpargeAttn and How

- Created: 2026-09-30
- Fork: `ussoewwin/SpargeAttn-hswq` (base: `thu-ml/SpargeAttn` @ `ae5b629`, "reduce repo size")
- Fork head: `119bba2` (tag `v1.0.0`)
- Package: `spas_sage_hswq_attn` (renamed from `spas_sage_attn` to distinguish from the official package)
- Diff size vs official: 23 files, +422 / -66 lines (of which the functional core is 3 files, ~370 lines)
- Design premise (Owner-mandated): **no quantization bit-width changes** — QK stays INT8, PV stays FP8. All improvements are calibration-accuracy / scale-selection / integration changes only.

---

## 1. What the official SpargeAttn does (baseline, for contrast)

Official SpargeAttn (ICML 2025, arXiv:2502.18137) accelerates attention with a two-stage online filter layered on top of SageAttention's quantized kernels:

1. **Stage-1 prediction** (`get_block_map_meansim_fuse_quant` in `utils.py`): Q and K are tiled into blocks; each block is mean-pooled into one representative token and its internal self-similarity (mean Gram-matrix value, `simthreshd1`) is measured. Highly self-similar blocks participate in a compressed attention map (`pooled_Q @ pooled_K^T`, softmaxed); `TopCdf(cdfthreshd)` or `TopK(topk)` selects which KV blocks matter per Q block. Non-similar blocks are "fix blocks" (always computed). The result is a binary block mask, converted to a LUT (`block_map_lut_triton`) consumed by the kernel.
2. **Stage-2 filter (kernel-side)**: inside the CUDA kernel (`qk_int_sv_f8_block_sparse_attn_kernel`), for each surviving KV block, after the INT8 `compute_int_qk`, `update_mo` computes `local_max_diff`; after warp/block reduction, `if (local_max_diff + pv_threshold > 0)` decides whether to run the expensive PV side (`RS_32_to_8` → `accumulate_d_f8` → `compute_fp8_sv_inst_buf*`). Skipped blocks cost only the INT8 QK MMA.
3. **Quantization**: Q/K per-block INT8 with scale = `max(|x|)/127 + 1e-7` (Q additionally folds `sm_scale*log2e` into the scale domain), V per-channel FP8 E4M3 (scale_max 2.25 for the fp16-accumulator path).
4. **Per-head hyperparameters**: `simthreshd1`, `cdfthreshd`/`topk`, `pvthreshd` — autotuned per layer/head by `SparseAttentionMeansim` (grid search + binary searches against fp16 SDPA ground truth with L1 gates).

## 2. What this fork changes

### 2.1 Package rename (identity, not behavior)

- Package directory and import name: `spas_sage_attn` → **`spas_sage_hswq_attn`**.
- Extension module names in `setup.py`: `spas_sage_hswq_attn._qattn`, `spas_sage_hswq_attn._fused` — so a built wheel installs under a distinct name and **cannot collide with the official `spas_sage_attn`** in the same Python environment (both can coexist, which matters because the live ComfyUI environment ships official SageAttention separately).
- API function names `spas_sage_attn_meansim[_topk]_cuda` → `spas_sage_hswq_attn_meansim[_topk]_cuda`. The `spas_sage2_*` / `block_sparse_sage2_*` function names are deliberately unchanged: those encode the technical classification ("based on SageAttention2"), while the package name now carries the fork identity.
- All internal imports, `evaluate/`, `inference_examples/`, Triton example and README import lines updated consistently (23 files).
- Version: `0.1.0` → **`1.0.0`**, tagged `v1.0.0`.

### 2.2 E2-a: per-block INT8 scale sweep (`spas_sage_hswq_attn/scale_sweep.py`, new, 154 lines)

**Problem**: official quantization always uses the max-abs scale, `max(|x|)/127 + 1e-7`, per block. This is optimal only when block values fill the range uniformly. Real attention blocks frequently have a single outlier (the max) plus a low-amplitude bulk; the max-abs scale then wastes most of the INT8 dynamic range on representing one value, inflating quantization error of everything else. Lower scales would represent the bulk more finely — at the cost of clipping the outlier. Which tradeoff wins is block-dependent, so the fork **measures instead of assumes**.

**Mechanism** (adapted from NVIDIA Model Optimizer's NVFP4 FP8 scale-sweep structure, `modelopt/torch/kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py` + `_fp8_scale_candidates.py`):

1. `int8_scale_candidates()`: 10 multiplicative ratios around the stock scale — `[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.1, 1.25, 1.5, 2.0]`. Ratio 1.0 is the stock scale itself, so the sweep can never do worse than stock up to float association noise.
2. `sweep_block_scales()`: for each block, quantize with each candidate scale, dequantize, and score the reconstruction MSE against the fp16 block; keep the minimum-error scale. Pure PyTorch (no Triton dependency), vectorized over blocks.
3. `per_block_int8_swept()`: drop-in replacement for `per_block_int8()` with **identical contract** — same input layout handling (HND), Q path folds `sm_scale*log2e` into the scale domain exactly as the official kernel does, output scale shape `(B, H, nblock, 1)`, the `+1e-7` epsilon kept so ratio-1.0 is bit-comparable to stock.
4. Integration in `core.py::spas_sage2_attn_meansim_topk_cuda` (the recommended API), env-gated:

```python
if swept_quant_enabled() and tensor_layout == "HND":
    try:
        q_int8, q_scale, k_int8, k_scale = per_block_int8_swept(
            q, k, BLKQ=128, BLKK=64, sm_scale=scale
        )
    except Exception:
        pass  # stock fused-quant tensors remain in place on any failure
```

Placement is deliberate: the block-map/LUT generation happens **before** this point and pools from the fp16 inputs, so the skip mask is unaffected by the chosen scales — only the tensors handed to the kernels change. Any failure inside the sweep falls back to the stock fused-quant tensors (fail-safe).

**Why this is bit-width preserving**: INT8 stays INT8. What changes is only which of the 10 scales is selected per block. Dequant math in the kernels (`out = int8 * q_scale * k_scale`) is untouched.

**Verified** (this machine, ComfyUI embedded python, torch CPU path):
- Q sweep MSE ≤ stock max-abs MSE on **100% of blocks** (256/256), in the kernel's folded domain (`sm_scale*log2e` applied to both sides).
- K sweep MSE ≤ stock on **100% of blocks** (256/256).
- Output contract (shapes/dtypes: `q_int8 (B,H,N,D) int8`, `q_scale (B,H,nb,1) f32`) matches `per_block_int8` exactly.
- Env gate default off; unset `SPARGE_SCALE_SWEEP` reproduces stock behavior byte-for-byte.

**Expected effect** (to be confirmed by the GPU evaluation): lower per-block quantization error → sharper, more faithful scores → the stage-1 prediction and stage-2 threshold operate on truer values. This can raise the realizable skip ratio at equal quality, or hold quality at higher sparsity. It does **not** change FLOPs; it improves the quality side of the quality/sparsity tradeoff that the hyperparameters control.

### 2.3 E2-b: threshold headroom calibration (`spas_sage_hswq_attn/headroom.py`, new, part 1)

**Problem**: the autotuned per-head hyperparameters (`simthreshd1`, `cdfthreshd`/`topk`, `pvthreshd`) are fitted on probe inputs to average behavior. A head whose score distribution has heavy tails (rare activation spikes) can sit right at the decision boundary: the autotuned threshold passes the quality gate on average but occasionally misses a spike, producing flicker-style artifacts that show up only on specific seeds/steps.

**Mechanism** (`calibrate_headroom()`):

1. Binary-search the **breaking point**: the `pvthreshd` value at which the layer's L1 error (vs fp16 SDPA ground truth, the same metric the official autotuner uses) first exceeds the quality gate `pv_l1`. Six iterations over `[0, base_pv]` (~1.6% resolution).
2. Compute the margin ratio `base_pv / breaking_pv` — how much slack the head actually has.
3. Apply a conservative multiplier (`target_gap`, default 1.10 = +10%) to the thresholds, **capped at 95% of the measured breaking point** so headroom can never push past observed failure:
   - `pvthreshd` ↑ (skip only when clearly safe)
   - `simthreshd1` more negative (more blocks classified "similar" → kept)
   - `topk` ↑ / `cdfthreshd` ↑ (more blocks kept in the mask)
4. Returns a dict; **does not write into the tuner** — the caller (HSWQ wrapper) owns storage, so the official `SparseAttentionMeansim` object stays stock.

Uses only inputs the official autotune already consumes (probe Q/K/V); no new calibration data.

### 2.4 E2-d: layer-importance weighting (`spas_sage_hswq_attn/headroom.py`, new, part 2)

**Problem**: official SpargeAttn treats every layer's quality budget identically. In diffusion UNets, layers are not equally sensitive — HSWQ's own DualMonitor sensitivity analysis already ranks them. Spending sparsity budget uniformly wastes compute on insensitive layers and starves sensitive ones.

**Mechanism** (`apply_importance_weighting()`): a config-level transform of the per-layer hyperparameters by an importance score in [0, 1] (normalized HSWQ sensitivity), with a bounded strength (`weight_strength`, default 0.5):

| Knob | importance → 1 (sensitive) | importance → 0 (insensitive) |
|---|---|---|
| `topk` | ↑ more blocks kept (≤ 1.0 clamp) | ↓ fewer blocks kept (≥ 0.05 clamp) |
| `cdfthreshd` | ↑ toward 1.0 | ↓ |
| `pvthreshd` | ↑ larger (skip only on clear safety) | ↓ |
| `simthreshd1` | ↓ more negative (more blocks pass the similarity test) | ↑ |

Direction checks verified: importance=1.0 raises topk (0.5 → 0.75 at strength 0.5) and pvthreshd, lowers simthreshd1; importance=0.0 lowers topk (0.5 → 0.25); all outputs clamped to valid ranges.

No kernel change — these are the same per-head tensors `hyperparameter_check()` already accepts as 1-D tensors, so per-layer values are natively supported.

### 2.5 What was NOT changed (and why)

| Candidate | Verdict | Reason |
|---|---|---|
| QK INT8 → NVFP4 | **Rejected** | Same structure as the measured SA3 failure (FP4 attention error accumulates across layers; NVFP4+SA3 cos 0.0556 in the HSWQ SA2 plan). The skip decision itself depends on QK score fidelity; FP4's 8-level mantissa granularity destroys it. |
| PV FP8 → FP4 | **On hold** | PV is a weighted sum (more FP4-tolerant than QK in principle), but the SA3 precedent puts the burden of proof on measurement; deferred until the stock evaluation lands. |
| Replacing the Triton/CUDA kernels | **Not needed** | SpargeAttn already fuses the skip into SageAttention2's quantized kernels — exactly the "INT8 QK + block skip + FP8 PV" combination. Re-implementing it would duplicate upstream work. |
| ModelOpt skip-softmax runtime / token merging | **Out of scope** | Different mechanism axis; competes with rather than composes with the in-kernel skip. |

## 3. Integration path into HSWQ (how the pieces connect)

1. Build the fork: `pip install ninja && python setup.py install` (CUDA ≥ 12.8 for Blackwell SM120; the wheel installs as `spas_sage_hswq_attn==1.0.0`, coexisting with the official `sageattention` package already in the live environment).
2. In `ComfyUI-HSWQ-Loader-and-Tools/hswq/hswq_sa2_accel.py::_make_attention_sage2`, the `sageattn` call is swapped for `spas_sage_hswq_attn.spas_sage2_attn_meansim_topk_cuda(q, k, v, topk=..., pvthreshd=..., return_sparsity=True)`. The existing pattern-resolution (checkpoint probe) machinery stays.
3. Per-layer hyperparameters come from the official autotuner (`SparseAttentionMeansim`), then pass through `calibrate_headroom()` and `apply_importance_weighting()` with HSWQ's DualMonitor importance values.
4. `SPARGE_SCALE_SWEEP=1` enables E2-a.
5. `return_sparsity=True` surfaces the realized QK sparsity per call, feeding the loader's status dump.

## 4. Verification ledger (what is measured vs pending)

| Item | Status |
|---|---|
| Q sweep MSE ≤ stock (folded domain), 256/256 blocks | **Measured, PASS** |
| K sweep MSE ≤ stock, 256/256 blocks | **Measured, PASS** |
| Output contract vs `per_block_int8` (shapes/dtypes) | **Measured, PASS** |
| Env gate default-off behavior | **Measured, PASS** |
| Importance weighting direction + clamps | **Measured, PASS** |
| Package syntax (all modules) | **Measured, PASS** |
| GPU end-to-end (kernel launch on SM120, SSIM, s/it, skip ratio) | **Pending** — requires `python setup.py install` build; not run yet |
| End-to-end quality/speed gates (12-step SSIM protocol) | **Pending** — after build + HSWQ wrapper swap |

## 5. File-level change map (official `ae5b629` → fork `119bba2`)

| File | Change |
|---|---|
| `spas_sage_hswq_attn/scale_sweep.py` | **NEW** (154 L): candidate generator, torch-reference sweep, drop-in wrapper, env gate |
| `spas_sage_hswq_attn/headroom.py` | **NEW** (190 L): headroom calibration, importance weighting, standalone-loadable fallback for `precision_metric` |
| `spas_sage_hswq_attn/core.py` | +2 import lines, +10 lines E2-a integration (gated, fail-safe) in `spas_sage2_attn_meansim_topk_cuda`; package-internal imports renamed |
| `spas_sage_hswq_attn/{__init__,autotune,utils,quant_per_block,quant_per_warp_cuda}.py` | Renamed from `spas_sage_attn/`; import lines updated; autotune import updated. **No logic changes** |
| `setup.py` | Package/extension names → `spas_sage_hswq_attn.*`; `version='1.0.0'` |
| `README.md`, `evaluate/*`, `inference_examples/*`, `Triton_SpargeAttn/*` | Import lines and install instructions updated to the new package name |
| Kernel sources (`csrc/**`) | **Unchanged** — official kernels used as-is |

## 6. Reference: ModelOpt techniques that fed the design

- `modelopt/torch/kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py` + `_fp8_scale_candidates.py` — the evaluate-N-candidates-in-one-pass structure that E2-a adapts (candidate set + min-error selection), re-targeted from NVFP4/FP8 scales to INT8 block scales.
- `modelopt/torch/quantization/calib/nvfp4_act_headroom.py` — the headroom concept (reserve margin so downstream saturation never occurs) that E2-b applies to skip thresholds.
- `modelopt/torch/quantization/utils/shared_input.py::find_shared_input_groups` — noted (E2-c, not yet implemented) for fused-QKV models where Q/K scales must stay consistent across consumers of the same weights.
- HSWQ DualMonitor sensitivity analysis — the importance source for E2-d.
- HSWQ SA2 plan (2026-09-10) — the SA3 measured failure (NVFP4+SA3 cos 0.0556) that rules out any FP4 in the QK path.
