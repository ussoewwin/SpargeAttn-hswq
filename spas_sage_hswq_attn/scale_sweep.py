"""
Scale sweep for SpargeAttn per-block INT8 quantization (bit-width preserving).

Adapted from NVIDIA Model Optimizer's NVFP4 FP8 scale-sweep structure
(modelopt/torch/kernels/quantization/gemm/nvfp4_fp8_scale_sweep.py +
_fp8_scale_candidates.py): instead of always using the max-based scale,
evaluate N candidate scales per block against the fp16 reference block
output and keep the minimum-error one. INT8 stays INT8; only which scale
is chosen changes.

Design notes (verified against SpargeAttn source):
- SpargeAttn quantizes per block in triton_bmm_pool_sim_simmean_fuse_quant
  (utils.py): scale = max(|x|)/127, i.e. the max-abs scale. This is the
  candidate generator's center point.
- The dequant path in the kernels is out = int8 * q_scale * k_scale, so a
  per-block scale tensor with the same layout is drop-in compatible.
- The first-stage prediction (get_block_map_meansim_fuse_quant) computes
  pooled means from the FP16 input, NOT from the quantized values, so
  changing weight scales does not perturb the skip mask generation.
- Quantized inputs only affect: QK^T scores (via q_scale/k_scale) and the
  PV path (V stays FP16 input in sage2 path; V is quantized to FP8 inside
  core.py with its own per-channel scale).

This module provides:
  1. int8_scale_candidates() - candidate generator (pure torch, no triton)
  2. sweep_block_scales()   - torch reference sweep (works without triton)
  3. per_block_int8_swept() - drop-in replacement for per_block_int8(),
     gated by an env var so stock behavior stays the default.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F


def int8_scale_candidates(device: torch.device | str = "cpu") -> torch.Tensor:
    """Generate INT8 scale candidate ratios around the max-abs scale.

    The candidate set is multiplicative around 1.0 (the stock max-abs scale).
    Ratios below 1.0 improve resolution for low-amplitude blocks (better
    int8 usage when the max is an outlier); ratios above 1.0 trade range for
    headroom. candidates[0] == 1.0 keeps the stock behavior as a candidate.
    """
    ratios = torch.tensor(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 1.1, 1.25, 1.5, 2.0],
        dtype=torch.float32,
        device=device,
    )
    return ratios


def _quantize_block_int8(x_fp32: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize x with the given scale, matching SpargeAttn kernel rounding:
    round half away from zero, clamp to int8 range.
    """
    x_scaled = x_fp32 / scale
    x_int8 = torch.sign(x_scaled) * torch.floor(torch.abs(x_scaled) + 0.5)
    return x_int8.clamp(-127, 127)


def sweep_block_scales(
    x: torch.Tensor,
    block_size: int,
    *,
    max_candidates: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch-reference scale sweep for per-block INT8 quantization.

    Args:
        x: FP16/BF16 input, (B, H, N, D) HND layout.
        block_size: block size along N (BLKQ for Q, BLKK for K).
        max_candidates: candidate count cap.

    Scoring: per-block dequantized MSE against the fp16 block. This is an
    intra-block proxy for score error; the attention-level (end-to-end)
    hyperparameter search remains in SparseAttentionMeansim autotune.

    Returns:
        (x_int8 int8 (B,H,N,D), x_scale float32 (B,H,nblock))
    """
    B, H, N, D = x.shape
    nblock = (N + block_size - 1) // block_size
    pad = nblock * block_size - N
    x_padded = F.pad(x, (0, 0, 0, pad)) if pad else x
    blocks = x_padded.reshape(B, H, nblock, block_size, D).float()

    max_abs = blocks.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    base_scale = max_abs / 127.0 + 1e-7  # stock scale incl. kernel epsilon (max|.|/127 + 1e-7)

    ratios = int8_scale_candidates(x.device)[:max_candidates]  # (C,)

    best_err = None
    best_scale = None
    for ci in range(ratios.numel()):
        s = base_scale * ratios[ci]  # (B,H,nb,1,1)
        q = _quantize_block_int8(blocks, s)
        err = ((q * s - blocks) ** 2).sum(dim=(-2, -1))  # (B,H,nb)
        s_h = s[:, :, :, 0, 0]  # (B,H,nb)
        if best_err is None:
            best_err = err
            best_scale = s_h
        else:
            improved = err < best_err
            best_err = torch.where(improved, err, best_err)
            best_scale = torch.where(improved, s_h, best_scale)

    s = best_scale.unsqueeze(-1).unsqueeze(-1)
    q = _quantize_block_int8(blocks, s)
    q_int8 = q.reshape(B, H, nblock * block_size, D).to(torch.int8)
    if pad:
        q_int8 = q_int8[:, :, :N, :]
    return q_int8, best_scale.to(torch.float32)


def per_block_int8_swept(
    q: torch.Tensor,
    k: torch.Tensor,
    BLKQ: int = 128,
    BLKK: int = 64,
    sm_scale: float | None = None,
    tensor_layout: str = "HND",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for quant_per_block.per_block_int8 with sweeping.

    Convention notes (verified against SpargeAttn source):
    - quant_per_block.py folds sm_scale*log2e into Q before max-abs
      (sm_scale=(sm_scale * 1.44269504)); replicated here.
    - per_block_int8 returns scales shaped (B, H, nblock, 1); replicated.
    """
    if tensor_layout != "HND":
        raise NotImplementedError("sweep path currently supports HND layout only")

    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    q_scaled = (q.float() * (sm_scale * 1.44269504)).to(q.dtype)
    q_int8, q_scale = sweep_block_scales(q_scaled, BLKQ)
    k_int8, k_scale = sweep_block_scales(k, BLKK)

    if q_scale.dim() == 3:
        q_scale = q_scale.unsqueeze(-1)
    if k_scale.dim() == 3:
        k_scale = k_scale.unsqueeze(-1)
    return q_int8, q_scale, k_int8, k_scale


def swept_quant_enabled() -> bool:
    """Env gate. SPARGE_SCALE_SWEEP=1 enables the swept quantization path."""
    return os.environ.get("SPARGE_SCALE_SWEEP", "").strip().lower() in (
        "1", "true", "on", "enable",
    )
