"""Student-written tests for the Your Turn extension logic (finops/*).

These cover the new functions only; the graded suite is untouched.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics, sustainability


# --- Ext 1: recommend_tier_v2 -------------------------------------------------
def test_tier_v2_low_churn_rides_spot():
    d = pricing.recommend_tier_v2(20, True, job_days=14, gpu_type="H100")
    assert d["tier"] == "spot"


def test_tier_v2_high_churn_leaves_spot():
    # L4 spot churns at 15%/hr -> a low-duty interruptible job should NOT be spot
    d = pricing.recommend_tier_v2(8, True, job_days=22, gpu_type="L4")
    assert d["tier"] != "spot"


def test_tier_v2_short_job_prefers_1yr_over_3yr():
    d = pricing.recommend_tier_v2(24, False, job_days=30, gpu_type="A100")
    assert d["tier"] == "reserved" and d["reserved_term"] == "1yr"


def test_tier_v2_long_job_gets_3yr():
    d = pricing.recommend_tier_v2(24, False, job_days=800, gpu_type="A100")
    assert d["reserved_term"] == "3yr"


# --- Ext 3: cache economics -------------------------------------------------
def test_cache_break_even_and_gate():
    be = pricing.cache_break_even_reads(write_cost_multiplier=3.0, read_discount=0.10)
    assert abs(be - (2.0 / 0.9)) < 1e-9
    assert pricing.cache_is_worth_it(be + 0.5, write_cost_multiplier=3.0) is True
    assert pricing.cache_is_worth_it(be - 0.5, write_cost_multiplier=3.0) is False


# --- Ext 2: right-sizing by bandwidth ------------------------------------------
def test_dollars_per_tbs_ranks_by_bandwidth_value():
    # A cheap, bandwidth-starved L4 is worse $/(TB/s) than an H100
    assert metrics.dollars_per_tbs(0.80, 0.30) > metrics.dollars_per_tbs(2.50, 3.35)


def test_rightsize_picks_cheaper_card_that_still_fits():
    cat = {
        "H100": {"on_demand_hr": 2.50, "peak_bw_tbs": 3.35},
        "A100": {"on_demand_hr": 1.79, "peak_bw_tbs": 2.00},
        "L4":   {"on_demand_hr": 0.80, "peak_bw_tbs": 0.30},
    }
    cur = [{"gpu_id": "g0", "gpu_type": "H100", "mbu": 0.20}]  # 0.67 TB/s achieved
    picks = metrics.rightsize_by_mbu(cur, cat)
    assert picks and picks[0]["to"] == "A100" and picks[0]["saved_per_hr"] > 0


# --- Ext 5: carbon-aware scheduling ------------------------------------------
def test_relocate_job_cuts_carbon():
    r = sustainability.relocate_job_carbon(1000.0, "us-east-1", "europe-north1")
    assert r["carbon_saved_g"] > 0 and 0 < r["carbon_cut_pct"] <= 100


def test_region_table_sorted_and_complete():
    t = sustainability.region_table()
    assert len(t) == len(sustainability.REGION_CARBON)
    assert t == sorted(t, key=lambda r: r["blended_score"])
