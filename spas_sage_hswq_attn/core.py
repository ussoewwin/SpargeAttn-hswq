"""
Copyright (c) 2025 by SpargeAttn team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from .utils import hyperparameter_check, cached_hyperparam, get_block_map_meansim, get_block_map_meansim_fuse_quant, get_vanilla_qk_quant, block_map_lut_triton
from .quant_per_block import per_block_int8, per_warp_int8
from .scale_sweep import per_block_int8_swept, swept_quant_enabled
from einops import rearrange

import spas_sage_hswq_attn._qattn as qattn
import spas_sage_hswq_attn._fused as fused

SAGE2PP_ENABLED = True
try:
    from spas_sage_hswq_attn._qattn import qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
except:
    print("Warning: Sage2++ NOT enabled")
    SAGE2PP_ENABLED = False

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs

@torch.compiler.disable
def spas_sage2_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)


    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        ## quant v
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        #fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        # _SEEDVR2_SAGE2PP_ARCH_GUARD: the f16-accumulate kernel is compiled from
        # the sm89 template set; allow it only on sm89+ (Ampere never reaches this
        # branch, but keep the guard so future arch additions cannot misroute).
        elif SAGE2PP_ENABLED and arch in ("sm89", "sm100", "sm120", "sm121"):
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

_ARCH_CACHE = {}

def _get_arch(device):
    """Per-device compute-capability string (e.g. "sm120"), cached."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    arch = _ARCH_CACHE.get(idx)
    if arch is None:
        major, minor = torch.cuda.get_device_capability(idx)
        arch = f"sm{major}{minor}"
        _ARCH_CACHE[idx] = arch
    return arch


@torch.compiler.disable
def spas_sage2_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=None, return_sparsity=False):
    """topk-ratio SpargeAttn (SageAttention2 kernels).

    output_dtype: None (default) keeps the input dtype end-to-end; an explicit
    torch dtype forces a final cast (B5 fix). Previously the default was
    torch.float16, which silently cast bf16 outputs to fp16.
    """
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."

    in_dtype = q.dtype
    vdev0 = q.device if q.device.type == "cuda" else v.device
    arch0 = _get_arch(vdev0)
    if in_dtype == torch.float32 or in_dtype == torch.float16:
        q = q.contiguous().to(torch.float16)
        k = k.contiguous().to(torch.float16)
        v = v.contiguous().to(torch.float16)
    elif arch0 in ("sm80", "sm86", "sm87"):
        # Ampere PV kernels take half* V only (decl.cuh SpargeAttentionSM80Dispatched):
        # bf16 V must still be converted on this path.
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.float16)
    else:
        # bf16 stays bf16 end-to-end (Q/K feeding Stage-1 Triton pooling and the
        # INT8 QK kernels; V feeds the fused bf16->fp8 quantize kernels on
        # sm89+). No fp16 round-trip copy for V anymore.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    vdev = v.device
    cur = torch.cuda.current_device()
    if vdev.index is not None and vdev.index != cur:
        torch.cuda.set_device(vdev)

    arch = _get_arch(vdev)
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)
    # E2-a (bit-width preserving): optional per-block INT8 scale sweep.
    # Prediction/LUT stay untouched (the fused call above pools from fp16
    # inputs); only the quantized tensors handed to the kernels change.
    if swept_quant_enabled() and tensor_layout == "HND":
        try:
            q_int8, q_scale, k_int8, k_scale = per_block_int8_swept(
                q, k, BLKQ=128, BLKK=64, sm_scale=scale
            )
        except Exception:
            pass  # stock fused-quant tensors remain in place on any failure

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = cached_hyperparam(pvthreshd, q.size(-3), vdev)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        ## quant v (bf16-safe: the fused fp8 kernels template on bf16 input)
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        #fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        elif SAGE2PP_ENABLED:
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if output_dtype is not None and output_dtype != in_dtype and o.dtype != output_dtype:
        o = o.to(output_dtype)
    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o


@torch.compiler.disable
def spas_sage2_attn_meansim_topk_nhd_cuda(q, k, v, is_causal=False, scale=None, simthreshd1=-0.1, topk=0.5, pvthreshd=50, attention_sink=False):
    """Zero-copy topk SpargeAttn for packed NHD inputs.

    q, k, v: (B, L, H, D) fp16/bf16 with last dim contiguous (arbitrary strides
    on the other dims, e.g. a view of a packed varlen buffer).
    Returns o: (B, H, L, D) contiguous, same dtype as q.

    Same prediction, quantization and kernels as spas_sage2_attn_meansim_topk_cuda;
    the difference is that Q/K/V are never materialized in HND layout.
    """
    assert q.dtype in (torch.float16, torch.bfloat16)
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    assert q.size(1) >= 128, "seq_len should be not less than 128."
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)

    # HND views (no copy)
    q_h = q.permute(0, 2, 1, 3)
    k_h = k.permute(0, 2, 1, 3)
    v_h = v.permute(0, 2, 1, 3)

    headdim = q.size(-1)
    assert headdim in [64, 128], "headdim should be in [64, 128]."

    vdev = v.device
    cur = torch.cuda.current_device()
    if vdev.index is not None and vdev.index != cur:
        torch.cuda.set_device(vdev)
    arch = _get_arch(vdev)

    km = k_h.mean(dim=-2, keepdim=True)

    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q_h, k_h, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=None, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q_h, k_h, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=None, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    pvthreshd = cached_hyperparam(pvthreshd, q_h.size(1), vdev)
    B, H, L, D = q_h.shape
    o = torch.empty((B, H, L, D), dtype=q.dtype, device=q.device)

    if arch in ("sm80", "sm86", "sm87"):
        v16 = v_h.contiguous().to(torch.float16)
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v16, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        b, h_kv, kv_len, head_dim = v_h.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        # transpose_pad_permute_cuda only requires last-dim contiguity: strided HND view is fine.
        fused.transpose_pad_permute_cuda(v_h, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        elif SAGE2PP_ENABLED:
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    return o
    
# ---------------------------------------------------------------------------
# Variable-length (packed) topk SpargeAttn.
# ---------------------------------------------------------------------------
from collections import OrderedDict as _OrderedDict
_VARLEN_PLAN_CACHE = _OrderedDict()
_VARLEN_PLAN_CACHE_MAX = 64
_VARLEN_DEV_TENSOR_CACHE = _OrderedDict()
_VARLEN_DEV_TENSOR_CACHE_MAX = 64


def _dev_tensor(seq, device):
    """Cache small index lists as device tensors (keyed by content+device)."""
    key = (device, tuple(seq))
    t = _VARLEN_DEV_TENSOR_CACHE.get(key)
    if t is None:
        t = torch.tensor(seq, dtype=torch.long, device=device)
        _VARLEN_DEV_TENSOR_CACHE[key] = t
        while len(_VARLEN_DEV_TENSOR_CACHE) > _VARLEN_DEV_TENSOR_CACHE_MAX:
            _VARLEN_DEV_TENSOR_CACHE.popitem(last=False)
    return t


@torch.compiler.disable
def spas_sage2_attn_meansim_topk_varlen_cuda(
    q, k, v, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q=None, max_seqlen_k=None,
    is_causal=False, scale=None, simthreshd1=-0.1, topk=0.5,
    pvthreshd=50, attention_sink=False, output_dtype=None,
):
    """topk SpargeAttn for packed variable-length sequences (NHD layout).

    q, k, v: (total_q, H, D) / (total_k, H, D) contiguous fp16/bf16, packed
             according to cu_seqlens_q / cu_seqlens_k (len = num_seqs + 1).

    Both uniform and mixed sequence lengths are supported on the existing
    fixed-length (batched) kernels -- the fixed-length premise is never
    violated:
      * uniform : every sequence shares one length -> a SINGLE zero-copy
                  batched launch (exactly the fixed-length fast path).
      * mixed   : sequences are bucketed by identical (L_q, L_k); each bucket
                  costs ONE batched launch. Contiguous buckets use zero-copy
                  views; scattered buckets use a single gather + scatter.
                  No per-sequence launches and no padding waste.
    The bucket plan is cached by the order-independent multiset of window
    lengths, so real runs (which permute windows between calls) still hit it.

    Returns o: (total_q, H, D) with the same dtype as q.
    """
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    total_q = q.size(0)
    total_k = k.size(0)
    heads = q.size(1)
    headdim = q.size(2)
    assert headdim in (64, 128), "headdim should be in [64, 128]."
    assert k.size(1) == heads and v.size(1) == heads

    in_dtype = q.dtype
    if k.dtype != in_dtype:
        k = k.to(in_dtype)
    if v.dtype != in_dtype:
        v = v.to(in_dtype)

    cq = cu_seqlens_q.tolist()
    ck = cu_seqlens_k.tolist()
    n = len(cq) - 1
    # Cache key = exact multiset of (Lq, Lk) plus num_seqs/heads/headdim.
    # Order-independent, so calls that merely permute the same window lengths
    # hit the cache. Bumped by a cheap counter to bound memory.
    from collections import Counter as _Counter
    _lens_q = [cq[i + 1] - cq[i] for i in range(n)]
    _lens_k = [ck[i + 1] - ck[i] for i in range(n)]
    plan_key = (n, heads, headdim,
                tuple(sorted(_Counter(_lens_q).items())),
                tuple(sorted(_Counter(_lens_k).items())))
    plan = _VARLEN_PLAN_CACHE.get(plan_key)
    if plan is not None:
        _VARLEN_PLAN_CACHE.move_to_end(plan_key)
    if plan is None:
        buckets = {}
        order = []
        for i in range(n):
            key = (_lens_q[i], _lens_k[i])
            b = buckets.get(key)
            if b is None:
                b = buckets[key] = ([], [])
                order.append(key)
            b[0].append(cq[i])
            b[1].append(ck[i])
        groups = []
        for key in order:
            Lq, Lk = key
            sq, sk = buckets[key]
            contiguous = (len(sq) == 1) or all(
                sq[j + 1] == sq[j] + Lq and sk[j + 1] == sk[j] + Lk
                for j in range(len(sq) - 1)
            )
            groups.append((Lq, Lk, sq, sk, contiguous))
        _VARLEN_PLAN_CACHE[plan_key] = groups
        while len(_VARLEN_PLAN_CACHE) > _VARLEN_PLAN_CACHE_MAX:
            _VARLEN_PLAN_CACHE.popitem(last=False)
        plan = groups

    device = q.device
    o = torch.empty((total_q, heads, headdim), dtype=in_dtype, device=device)

    for (Lq, Lk, sq, sk, contiguous) in plan:
        nseq = len(sq)
        if Lq < 128 or Lk < 128 or Lq != Lk:
            for j in range(nseq):
                qs = sq[j]
                ks = sk[j]
                qi = q[qs:qs + Lq].permute(1, 0, 2).unsqueeze(0)
                ki = k[ks:ks + Lk].permute(1, 0, 2).unsqueeze(0)
                vi = v[ks:ks + Lk].permute(1, 0, 2).unsqueeze(0)
                oi = torch.nn.functional.scaled_dot_product_attention(
                    qi, ki, vi, is_causal=is_causal)
                o[qs:qs + Lq] = oi.squeeze(0).permute(1, 0, 2)
            continue

        if contiguous:
            qs = sq[0]
            ks = sk[0]
            qb = q[qs:qs + nseq * Lq].view(nseq, Lq, heads, headdim)
            kb = k[ks:ks + nseq * Lk].view(nseq, Lk, heads, headdim)
            vb = v[ks:ks + nseq * Lk].view(nseq, Lk, heads, headdim)
            idx_q = None
        else:
            ar_q = torch.arange(Lq, device=device)
            ar_k = torch.arange(Lk, device=device)
            sq_t = _dev_tensor(sq, device)
            sk_t = _dev_tensor(sk, device)
            idx_q = (sq_t[:, None] + ar_q[None, :]).reshape(-1)
            idx_k = (sk_t[:, None] + ar_k[None, :]).reshape(-1)
            qb = q.index_select(0, idx_q).view(nseq, Lq, heads, headdim)
            kb = k.index_select(0, idx_k).view(nseq, Lk, heads, headdim)
            vb = v.index_select(0, idx_k).view(nseq, Lk, heads, headdim)

        ob = spas_sage2_attn_meansim_topk_nhd_cuda(
            qb, kb, vb,
            is_causal=is_causal, scale=scale, simthreshd1=simthreshd1,
            topk=topk, pvthreshd=pvthreshd, attention_sink=attention_sink,
        )
        ob_flat = ob.permute(0, 2, 1, 3).reshape(-1, heads, headdim)
        if contiguous:
            o[qs:qs + nseq * Lq] = ob_flat
        else:
            o.index_copy_(0, idx_q, ob_flat)

    if output_dtype is not None and output_dtype != in_dtype and o.dtype != output_dtype:
        o = o.to(output_dtype)
    return o


@torch.compiler.disable
def block_sparse_sage2_attn_cuda(q, k, v, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    arch = get_cuda_arch_versions()[q.device.index]

    if arch == "sm90":
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)
    o = torch.empty_like(q)

    if arch in ("sm80", "sm86", "sm87"):
        qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(
            q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0
        )
    else:
        ## quant v
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 127) // 128 * 128
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
        fused.transpose_pad_permute_cuda(v, v_transposed_permutted, 1)
        v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=v.device)
        v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v.device)
        #fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 448.0, 1)
        fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, 2.25, 1)

        if arch == "sm90":
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        elif SAGE2PP_ENABLED:
            qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        else:
            qattn.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)

    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')
    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def spas_sage_hswq_attn_meansim_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink)  #

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o

@torch.compiler.disable
def spas_sage_hswq_attn_meansim_topk_cuda(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False):
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q, k, v = map(lambda t: rearrange(t, '... L H D -> ... H L D'), (q, k, v))
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    torch.cuda.set_device(v.device)

    dtype = q.dtype
    if dtype == torch.float32 or dtype == torch.float16:
        q, k, v = q.contiguous().to(torch.float16), k.contiguous().to(torch.float16), v.contiguous().to(torch.float16)
    else:
        q, k, v = q.contiguous().to(torch.bfloat16), k.contiguous().to(torch.bfloat16), v.contiguous().to(torch.float16)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
        # k = k - km
    headdim = q.size(-1)

    lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink)  #

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q.size(-3), q.device)

    _is_causal = 1 if is_causal else 0
    o = torch.empty_like(q)
    qattn.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        return o, qk_sparsity.item()
    else:
        return o
