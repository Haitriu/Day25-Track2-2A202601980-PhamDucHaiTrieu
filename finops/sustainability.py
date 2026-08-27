"""Sustainability economics — energy and carbon as governed cost levers (deck §11).

Region selection cuts $ and carbon together; reasoning queries are an energy bomb.
"""
from __future__ import annotations

# Grid carbon intensity (gCO2 / kWh) — illustrative 2026 snapshot.
REGION_CARBON = {
    "us-east-1": 380,
    "us-west-2": 120,   # Oregon hydro
    "europe-north1": 30,  # Norway
    "europe-central2": 660,  # Poland (dirtiest)
    "us-east-wa": 90,
}
# Electricity price (USD / kWh) — illustrative.
REGION_PRICE_KWH = {
    "us-east-1": 0.12,
    "us-west-2": 0.07,
    "europe-north1": 0.09,
    "europe-central2": 0.18,
    "us-east-wa": 0.055,
}

REASONING_ENERGY_MULTIPLIER = 80.0  # deck: reasoning ~74-86x a small-model query


def wh_per_query(total_tokens: int, wh_per_1k_tokens: float = 0.30, is_reasoning: bool = False) -> float:
    """Energy for one query. Median Gemini prompt ~0.24 Wh; reasoning ~74-86x."""
    base = (total_tokens / 1000.0) * wh_per_1k_tokens
    return base * (REASONING_ENERGY_MULTIPLIER if is_reasoning else 1.0)


def carbon_g(wh: float, region: str = "us-east-1") -> float:
    """Grams CO2 for an energy amount in a region."""
    gco2_kwh = REGION_CARBON.get(region, 400)
    return (wh / 1000.0) * gco2_kwh


def energy_cost_usd(wh: float, region: str = "us-east-1") -> float:
    """Electricity cost of an energy amount in a region."""
    return (wh / 1000.0) * REGION_PRICE_KWH.get(region, 0.12)


def tokens_per_watt(total_tokens: int, wh: float, seconds: float = 1.0) -> float:
    """Energy efficiency of serving: tokens per watt (higher is better)."""
    watts = (wh * 3600.0) / seconds if seconds > 0 else 0.0
    return total_tokens / watts if watts > 0 else 0.0


# ---------------------------------------------------------------------------
# Extension 5 — carbon-aware scheduling for interruptible jobs
# ---------------------------------------------------------------------------

def region_table() -> list:
    """One row per region with price, carbon and a blended rank (lower is better)."""
    rows = []
    max_p = max(REGION_PRICE_KWH.values())
    max_c = max(REGION_CARBON.values())
    for region in REGION_CARBON:
        p = REGION_PRICE_KWH.get(region, 0.12)
        c = REGION_CARBON[region]
        rows.append({
            "region": region,
            "usd_per_kwh": p,
            "gco2_per_kwh": c,
            "blended_score": round(0.5 * p / max_p + 0.5 * c / max_c, 3),
        })
    return sorted(rows, key=lambda r: r["blended_score"])


def cheapest_region() -> str:
    return min(REGION_PRICE_KWH, key=REGION_PRICE_KWH.get)


def cleanest_region() -> str:
    return min(REGION_CARBON, key=REGION_CARBON.get)


def relocate_job_carbon(job_kwh: float, from_region: str = "us-east-1", to_region: str | None = None) -> dict:
    """gCO2e and $ saved by moving one job's energy draw to a cleaner region."""
    to_region = to_region or cleanest_region()
    from_c = carbon_g(job_kwh * 1000.0, from_region)
    to_c = carbon_g(job_kwh * 1000.0, to_region)
    from_usd = energy_cost_usd(job_kwh * 1000.0, from_region)
    to_usd = energy_cost_usd(job_kwh * 1000.0, to_region)
    return {
        "from_region": from_region,
        "to_region": to_region,
        "carbon_saved_g": round(from_c - to_c, 1),
        "carbon_cut_pct": round((1 - to_c / from_c) * 100, 1) if from_c else 0.0,
        "energy_usd_saved": round(from_usd - to_usd, 4),
    }
