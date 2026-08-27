"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


# ---------------------------------------------------------------------------
# Extension 2 — right-sizing memory-bound GPUs by MBU, not by $/GPU-hr
# ---------------------------------------------------------------------------

def dollars_per_tbs(on_demand_hr: float, peak_bw_tbs: float) -> float:
    """$/hr per TB/s of HBM bandwidth — the real unit price for a decode box.

    Memory-bound inference is bottlenecked on HBM bandwidth, so the honest
    price-per-capability is $/(TB/s), not $/GPU-hr. A cheap card with tiny
    bandwidth can be *more* expensive per unit of useful work.
    """
    if peak_bw_tbs <= 0:
        return float("inf")
    return on_demand_hr / peak_bw_tbs


def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """$/hr per GB of HBM — the sizing unit when a model must simply fit."""
    if hbm_gb <= 0:
        return float("inf")
    return on_demand_hr / hbm_gb


def rightsize_by_mbu(current, catalog, mbu_key: str = "mbu"):
    """For each memory-bound GPU, find the cheapest catalog card that still
    covers its *achieved* bandwidth demand, and report the $/hr delta.

    `current`  : iterable of dicts with 'gpu_id', 'gpu_type', mbu_key.
    `catalog`  : dict gpu_type -> dict with 'on_demand_hr', 'peak_bw_tbs'.
    Returns a list of {gpu_id, from, to, achieved_tbs, saved_per_hr}.
    """
    picks = []
    for row in current:
        cur_type = row["gpu_type"]
        if cur_type not in catalog:
            continue
        cur_bw = float(catalog[cur_type]["peak_bw_tbs"])
        cur_price = float(catalog[cur_type]["on_demand_hr"])
        # Bandwidth the workload actually pulls right now.
        demand_tbs = float(row.get(mbu_key, 0.0)) * cur_bw
        best_type, best_price = cur_type, cur_price
        for gtype, c in catalog.items():
            bw = float(c["peak_bw_tbs"])
            price = float(c["on_demand_hr"])
            # keep 20% headroom over current demand
            if bw >= demand_tbs * 1.2 and price < best_price:
                best_type, best_price = gtype, price
        if best_type != cur_type:
            picks.append({
                "gpu_id": row["gpu_id"],
                "from": cur_type,
                "to": best_type,
                "achieved_tbs": round(demand_tbs, 3),
                "saved_per_hr": round(cur_price - best_price, 3),
            })
    return picks
