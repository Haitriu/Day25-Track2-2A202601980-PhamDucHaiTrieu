"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


# ---------------------------------------------------------------------------
# Extension 1 — smarter tier policy (interruption rate + 1yr vs 3yr horizon)
# ---------------------------------------------------------------------------

# Per-GPU-class per-hour spot reclamation probability (illustrative 2026).
# Big, in-demand accelerators churn less on the neoclouds than small ones.
SPOT_INTERRUPT_RATE = {
    "H100": 0.02, "H200": 0.03, "B200": 0.04, "MI300X": 0.05,
    "A100": 0.05, "A10G": 0.12, "L4": 0.15,
}


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    job_days: float,
    gpu_type: str | None = None,
    reserved_1yr_discount: float = 0.20,
    reserved_3yr_discount: float = 0.45,
    spot_effective_discount: float = 0.40,
    interrupt_rate: float | None = None,
    high_churn_threshold: float = 0.10,
) -> dict:
    """Return a tier decision *and* its reasoning, unlike the one-word v1.

    Two factors v1 ignores:
      1. Interruption rate varies by GPU class. A 15%/hr reclaim rate on an L4
         means near-constant rework — spot stops being a free lunch, so a
         high-churn interruptible job is pushed to on-demand/reserved instead.
      2. Reserved horizon. A 3-year commit only beats a 1-year commit if the
         job actually lives long enough to amortise it. Short jobs that clear
         the break-even duty cycle get 1yr; long ones get 3yr.
    """
    rate = interrupt_rate if interrupt_rate is not None else SPOT_INTERRUPT_RATE.get(gpu_type or "", 0.05)
    duty = max(0.0, hours_per_day) / 24.0
    be_3yr = break_even_utilization(reserved_3yr_discount)
    be_1yr = break_even_utilization(reserved_1yr_discount)

    if interruptible and hours_per_day < 24:
        if rate <= high_churn_threshold:
            return {"tier": "spot", "reserved_term": None,
                    "reason": f"interruptible, low spot churn ({rate:.0%}/hr) -> ride the discount"}
        # Too much rework to trust spot. Fall through on duty cycle.
        if duty >= be_3yr:
            term = "3yr" if job_days >= 365 else "1yr"
            return {"tier": "reserved", "reserved_term": term,
                    "reason": f"interruptible but high spot churn ({rate:.0%}/hr); steady duty {duty:.0%} -> reserved {term}"}
        return {"tier": "on_demand", "reserved_term": None,
                "reason": f"interruptible but high spot churn ({rate:.0%}/hr) and low duty {duty:.0%} -> on-demand"}

    if duty >= be_3yr:
        term = "3yr" if job_days >= 365 else "1yr"
        disc = reserved_3yr_discount if term == "3yr" else reserved_1yr_discount
        return {"tier": "reserved", "reserved_term": term,
                "reason": f"steady duty {duty:.0%} >= break-even {be_3yr:.0%}; horizon {job_days:.0f}d -> reserved {term} (-{disc:.0%})"}
    if duty >= be_1yr:
        return {"tier": "reserved", "reserved_term": "1yr",
                "reason": f"duty {duty:.0%} clears 1yr break-even {be_1yr:.0%} but not 3yr -> reserved 1yr"}
    return {"tier": "on_demand", "reserved_term": None,
            "reason": f"spiky / low duty {duty:.0%} -> on-demand"}


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_multiplier: float = 1.25,
    read_discount: float = 0.10,
) -> bool:
    """Extension 3 — does prompt caching actually pay for a given prefix?

    Writing a cache entry costs `write_cost_multiplier` x the normal input price
    (a one-time surcharge). Each subsequent read is billed at `read_discount` x
    price instead of 1.0x, saving (1 - read_discount) per read.

    Net vs. never-caching, for N reads of a prefix of size 1 (normalised):
        no-cache  : N * 1.0
        cache     : write_cost_multiplier + N * read_discount
    Cache wins when  N * (1 - read_discount) > (write_cost_multiplier - 1.0)
    """
    savings_per_read = 1.0 - read_discount
    write_surcharge = write_cost_multiplier - 1.0
    if savings_per_read <= 0:
        return False
    return avg_cache_reads * savings_per_read > write_surcharge


def cache_break_even_reads(write_cost_multiplier: float = 1.25, read_discount: float = 0.10) -> float:
    """Minimum number of reads per cached prefix for caching to break even."""
    savings_per_read = 1.0 - read_discount
    if savings_per_read <= 0:
        return float("inf")
    return (write_cost_multiplier - 1.0) / savings_per_read


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
