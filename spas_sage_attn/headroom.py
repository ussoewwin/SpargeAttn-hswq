"""
Threshold headroom calibration + importance weighting for SpargeAttn
(bit-width preserving; E2-b and E2-d of the v2 improvement plan).

E2-b (headroom): the autotuned per-head hyperparameters (simthreshd1,
cdfthreshd/topk, pvthreshd) are fitted to average behavior. Blocks whose
score distribution has heavy tails (rare spikes) sit right at the decision
boundary, causing occasional missed-spike artifacts. A per-head HEADROOM
margin widens the computed region: effectively the skip decisions become
more conservative for heads whose observed worst-case error is close to the
quality gate. Headroom is calibrated from the same probe inputs autotune
already uses (no new data), by re-running the worst candidate configuration
and measuring the L1 gap between the accepted and next-rejected threshold.

E2-d (importance weighting): HSWQ provides per-layer importance scores
(DualMonitor sensitivity analysis). Layers with high importance get a
more conservative (smaller) effective topk / larger pvthreshd margin, so
compute concentrates on sensitive layers. Implemented as a config-level
multiplier applied at call time in the HSWQ wrapper - no kernel change.

Both mechanisms are pure-config: they only move the per-head
hyperparameter tensors (simthreshd1 / cdfthreshd / topk / pvthreshd)
that SpargeAttn already supports as tensors via hyperparameter_check().
"""

from __future__ import annotations

import torch

try:
    from .utils import precision_metric
except ImportError:  # standalone load (no package context)
    def precision_metric(quant_o, fa2_o, verbose=True, round_num=4):
        import torch.nn.functional as F
        x, xx = quant_o.float(), fa2_o.float()
        sim = F.cosine_similarity(x.reshape(1, -1), xx.reshape(1, -1)).item()
        l1 = ((x - xx).abs().sum() / xx.abs().sum()).item()
        rmse = torch.sqrt(torch.mean((x - xx) ** 2)).item()
        return {"Cossim": round(sim, round_num), "L1": round(l1, round_num), "RMSE": round(rmse, round_num)}


@torch.no_grad()
def calibrate_headroom(
    tuner,  # SparseAttentionMeansim instance (already autotuned per head)
    qi: torch.Tensor,
    ki: torch.Tensor,
    vi: torch.Tensor,
    head_idx: int,
    *,
    mask=None,
    is_causal: bool = False,
    smooth_k: bool = True,
    probe_count: int = 2,
    target_gap: float = 1.10,
) -> dict:
    """Calibrate a per-head safety margin by probing threshold sensitivity.

    Args:
        tuner: SparseAttentionMeansim with autotuned hyperparams for head_idx.
        qi/ki/vi: probe tensors for this layer/head, (1, 1, N, D) or batched
            with head dim already split (same convention as autotune()).
        head_idx: head index into tuner's per-head parameter tensors.
        probe_count: number of alternative probe inputs to test robustness
            (calls re-use the same tensors; HSWQ passes 2 different-noise
            variants when available).
        target_gap: the multiplier applied to the autotuned thresholds.
            1.10 = 10% more conservative.

    Returns:
        dict with the adjusted hyperparams for this head:
        {'simthreshd1', 'cdfthreshd', 'topk' or None, 'pvthreshd'}
        Values are NOT written into tuner; the caller (HSWQ wrapper) owns
        where they are stored, keeping the tuner stock.
    """
    base_sim = float(tuner.simthreshd1[head_idx])
    base_pv = float(tuner.pvthreshd[head_idx])
    base_cdf = float(tuner.cdfthreshd[head_idx]) if tuner.cdfthreshd is not None else None
    base_topk = float(tuner.topk[head_idx]) if getattr(tuner, "topk", None) is not None else None

    # Measure L1 sensitivity around the operating point by nudging pvthreshd
    # down until quality breaks: the gap between the autotuned threshold and
    # the break point tells how much margin the head actually has.
    gt = torch.nn.functional.scaled_dot_product_attention(
        qi, ki, vi, mask, is_causal=is_causal
    )
    kernel = tuner.kernel_selection()

    def l1_at(pv: float) -> float:
        sparse_i, _ = kernel(
            qi, ki, vi, mask,
            is_causal=is_causal,
            smooth_k=smooth_k,
            cdfthreshd=base_cdf if base_topk is None else None,
            topk=base_topk if base_topk is not None else None,
            simthreshd1=base_sim,
            pvthreshd=pv,
            return_sparsity=False,
        )
        return precision_metric(sparse_i, gt, verbose=False)["L1"]

    pv_l1_gate = float(tuner.pv_l1)
    # Binary search the breaking pvthreshd (where L1 exceeds the gate)
    lo, hi = 0.0, base_pv
    breaking = None
    for _ in range(6):  # ~1.6% resolution over [0, base_pv]
        mid = (lo + hi) / 2
        if l1_at(mid) < pv_l1_gate:
            lo = mid
        else:
            breaking = mid
            hi = mid
    if breaking is None:
        # threshold never broke: head is robust, minimal headroom needed
        margin_ratio = target_gap
    else:
        # distance from operating point to failure, normalized
        margin_ratio = max(1.0, (base_pv / max(breaking, 1e-6)))

    # Headroom: scale thresholds conservatively, capped so we never exceed
    # the breaking point when we know it.
    adj_pv = base_pv * target_gap
    if breaking is not None:
        adj_pv = min(adj_pv, breaking * 0.95)
    adj_pv = max(adj_pv, base_pv)  # headroom only ever widens the computed region

    # simthreshd1: more negative = more blocks classified similar = more
    # conservative prediction (keeps blocks). Apply the same conservatism.
    adj_sim = base_sim - abs(base_sim) * (target_gap - 1.0) - 1e-3

    # cdfthreshd higher = more blocks kept; topk higher = more blocks kept.
    adj_cdf = None if base_cdf is None else min(1.0, base_cdf + (1.0 - base_cdf) * (target_gap - 1.0))
    adj_topk = None if base_topk is None else min(1.0, base_topk * target_gap)

    return {
        "simthreshd1": adj_sim,
        "cdfthreshd": adj_cdf,
        "topk": adj_topk,
        "pvthreshd": adj_pv,
        "margin_ratio": margin_ratio,
    }


def apply_importance_weighting(
    hyperparams: dict,
    importance: float,
    *,
    weight_strength: float = 0.5,
) -> dict:
    """E2-d: adjust per-layer hyperparams by the layer's importance score.

    Args:
        hyperparams: {'simthreshd1','cdfthreshd','topk','pvthreshd'} (per head
            or per layer scalars; see calibrate_headroom).
        importance: layer importance in [0, 1] (HSWQ DualMonitor sensitivity,
            normalized). Higher = more important = compute more.
        weight_strength: how strongly importance moves the knobs, in [0, 1].
            0.5 = at most 50% shift from the calibrated value toward
            conservative (important) or aggressive (unimportant).

    Returns:
        Adjusted copy of hyperparams. Direction per knob:
        - topk / cdfthreshd: higher importance -> MORE blocks kept
        - pvthreshd: higher importance -> LARGER (skip only when clearly safe)
        - simthreshd1: higher importance -> MORE negative (more similar blocks)
    """
    w = max(0.0, min(1.0, weight_strength))
    imp = max(0.0, min(1.0, importance))
    # shift in [-w, +w]: +w for importance=1 (conservative), -w for 0 (aggressive)
    shift = (imp - 0.5) * 2 * w

    out = dict(hyperparams)

    if out.get("topk") is not None:
        tk = float(out["topk"])
        out["topk"] = min(1.0, max(0.05, tk + shift * tk))

    if out.get("cdfthreshd") is not None:
        cd = float(out["cdfthreshd"])
        out["cdfthreshd"] = min(1.0, max(0.05, cd + shift * (1.0 - cd)))

    if out.get("pvthreshd") is not None:
        pv = float(out["pvthreshd"])
        out["pvthreshd"] = max(0.0, pv + shift * pv * 0.5)

    if out.get("simthreshd1") is not None:
        sm = float(out["simthreshd1"])
        # more negative = keep more blocks
        out["simthreshd1"] = sm - abs(sm) * shift * 0.5 - shift * 1e-3

    return out
