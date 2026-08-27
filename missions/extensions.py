"""Your Turn extensions — all five, each with a measured before/after.

Run: python missions/extensions.py   ->  prints results + writes outputs/extensions.md

Nothing here is required for verify.py / pytest to pass; this is the "learn
deeper" layer. Each function returns a plain dict so m5_report.py can fold a
summary into outputs/report.md.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import load_csv, num, catalog_by_type, ROOT
from missions.m2_inference_levers import MODEL_PRICES
from finops import pricing, metrics, sustainability

DAYS = 30


# ---------------------------------------------------------------------------
# Extension 1 — smarter recommend_tier (interruption rate + 1yr vs 3yr)
# ---------------------------------------------------------------------------
def ext1_tier_policy() -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()

    def price_job(j, tier, term):
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        if tier == "spot":
            return pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        if tier == "reserved":
            rate = num(c["reserved_1yr_hr"]) if term == "1yr" else num(c["reserved_3yr_hr"])
            return gpu_hours * rate
        return gpu_hours * od

    v1_total = v2_total = od_total = 0.0
    table = []
    for j in jobs:
        gtype = j["gpu_type"]
        hpd = num(j["hours_per_day"])
        jdays = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        ngpu = int(num(j["num_gpus"]))
        od_cost = hpd * DAYS * ngpu * num(cat[gtype]["on_demand_hr"])

        v1 = pricing.recommend_tier(hpd, interruptible)
        v2 = pricing.recommend_tier_v2(hpd, interruptible, job_days=jdays, gpu_type=gtype)

        v1_cost = price_job(j, v1, "3yr")
        v2_cost = price_job(j, v2["tier"], v2["reserved_term"] or "3yr")
        v1_total += v1_cost
        v2_total += v2_cost
        od_total += od_cost
        table.append({
            "job_id": j["job_id"], "gpu": gtype,
            "v1": v1, "v2": f"{v2['tier']}" + (f"/{v2['reserved_term']}" if v2["reserved_term"] else ""),
            "v1_cost": round(v1_cost), "v2_cost": round(v2_cost), "why": v2["reason"],
        })

    return {
        "table": table,
        "v1_savings_pct": round((1 - v1_total / od_total) * 100, 1),
        "v2_savings_pct": round((1 - v2_total / od_total) * 100, 1),
        "v1_monthly": round(v1_total), "v2_monthly": round(v2_total),
        "on_demand_monthly": round(od_total),
    }


# ---------------------------------------------------------------------------
# Extension 2 — right-size memory-bound GPUs by MBU, not $/GPU-hr
# ---------------------------------------------------------------------------
def ext2_rightsize_mbu() -> dict:
    from missions import m1_efficiency_audit
    r1 = m1_efficiency_audit.run(verbose=False)
    raw = {r[0]: r for r in [
        ("H100", 2.50, 80, 3.35), ("H200", 3.95, 141, 4.80), ("A100", 1.79, 80, 2.00),
        ("A10G", 1.00, 24, 0.60), ("L4", 0.80, 24, 0.30), ("B200", 5.09, 192, 8.00),
        ("MI300X", 1.95, 192, 5.30),
    ]}
    cat = {k: {"on_demand_hr": v[1], "hbm_gb": v[2], "peak_bw_tbs": v[3]} for k, v in raw.items()}

    # memory-bound == inference GPUs whose MBU is low (they are bandwidth-starved,
    # so they are over-specced on FLOPs they never use).
    mem_bound = [s for s in r1["summary"] if s["mbu"] < 0.35 and s["gpu_type"] in ("H100", "H200", "A100")]
    picks = metrics.rightsize_by_mbu(mem_bound, cat)
    monthly_saved = sum(p["saved_per_hr"] for p in picks) * 24 * DAYS

    price_per_tbs = {g: round(metrics.dollars_per_tbs(cat[g]["on_demand_hr"], cat[g]["peak_bw_tbs"]), 3) for g in cat}
    return {
        "candidates": mem_bound and [s["gpu_id"] for s in mem_bound],
        "picks": picks,
        "monthly_saved": round(monthly_saved),
        "price_per_tbs": price_per_tbs,
    }


# ---------------------------------------------------------------------------
# Extension 3 — cache_is_worth_it(): break-even reads vs the real dataset
# ---------------------------------------------------------------------------
def ext3_cache_economics() -> dict:
    rows = load_csv("token_usage.csv")
    cached_reqs = [r for r in rows if int(num(r["cached_input_tokens"])) > 0]
    # teams that share one big static system prompt -> ~1 cached prefix per team
    caching_teams = {r["team"] for r in cached_reqs}
    n_prefixes = max(1, len(caching_teams))
    avg_reads = len(cached_reqs) / n_prefixes

    be = pricing.cache_break_even_reads()
    return {
        "cached_requests": len(cached_reqs),
        "distinct_prefixes_est": n_prefixes,
        "avg_reads_per_prefix": round(avg_reads, 1),
        "break_even_reads": round(be, 2),
        "worth_it": pricing.cache_is_worth_it(avg_reads),
        "worth_it_at_be_minus_1": pricing.cache_is_worth_it(be - 1),
    }


# ---------------------------------------------------------------------------
# Extension 4 — reasoning budget: $ and Wh split, and a cap proposal
# ---------------------------------------------------------------------------
def ext4_reasoning_budget() -> dict:
    rows = load_csv("token_usage.csv")
    def bucket(pred):
        cost = wh = toks = n = 0
        for r in rows:
            if not pred(r):
                continue
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            pin, pout = MODEL_PRICES[r["route_tier"]]
            cost += pricing.request_cost(inp, out, pin, pout,
                                         cached_in=int(num(r["cached_input_tokens"])),
                                         batch=bool(int(num(r["is_batch"]))))
            wh += sustainability.wh_per_query(inp + out, is_reasoning=bool(int(num(r["is_reasoning"]))))
            toks += inp + out
            n += 1
        return {"requests": n, "cost": round(cost, 2), "wh": round(wh, 1), "tokens": toks}

    reasoning = bucket(lambda r: int(num(r["is_reasoning"])) == 1)
    normal = bucket(lambda r: int(num(r["is_reasoning"])) == 0)
    total_cost = reasoning["cost"] + normal["cost"]
    total_wh = reasoning["wh"] + normal["wh"]
    total_n = reasoning["requests"] + normal["requests"]

    # Proposal: a complexity gate that re-routes the reasoning requests that did
    # not need it. Assume ~60% of current reasoning traffic is over-triggered
    # (industry rule of thumb for un-gated "always think" configs). Those drop
    # the 6x output inflation and the ~80x energy multiplier.
    cur_share = reasoning["requests"] / total_n if total_n else 0
    over_triggered = 0.60
    cost_saved = reasoning["cost"] * over_triggered * (1 - 1.0 / 6.0)
    wh_saved = reasoning["wh"] * over_triggered * (1 - 1.0 / sustainability.REASONING_ENERGY_MULTIPLIER)

    return {
        "reasoning": reasoning, "normal": normal,
        "reasoning_traffic_pct": round(cur_share * 100, 1),
        "reasoning_cost_pct": round(reasoning["cost"] / total_cost * 100, 1) if total_cost else 0,
        "reasoning_energy_pct": round(reasoning["wh"] / total_wh * 100, 1) if total_wh else 0,
        "gate_cost_saved": round(cost_saved, 2),
        "gate_wh_saved": round(wh_saved, 1),
        "energy_multiple": round(reasoning["wh"] / reasoning["requests"] /
                                 (normal["wh"] / normal["requests"]), 1) if normal["requests"] and reasoning["requests"] else 0,
    }


# ---------------------------------------------------------------------------
# Extension 5 — carbon-aware scheduling of interruptible jobs
# ---------------------------------------------------------------------------
def ext5_carbon_schedule() -> dict:
    jobs = load_csv("workloads.csv")
    watts = {"H100": 700, "H200": 700, "A100": 400, "A10G": 150, "L4": 72, "B200": 1000, "MI300X": 750}
    rows = []
    total_saved_g = 0.0
    total_usd_saved = 0.0
    for j in jobs:
        if int(num(j["interruptible"])) != 1:
            continue
        gtype = j["gpu_type"]
        kwh = int(num(j["num_gpus"])) * num(j["hours_per_day"]) * num(j["days"]) * watts.get(gtype, 500) / 1000.0
        rel = sustainability.relocate_job_carbon(kwh, "us-east-1", sustainability.cleanest_region())
        total_saved_g += rel["carbon_saved_g"]
        total_usd_saved += rel["energy_usd_saved"]
        rows.append({"job_id": j["job_id"], "gpu": gtype, "kwh": round(kwh),
                     "carbon_saved_kg": round(rel["carbon_saved_g"] / 1000, 1),
                     "carbon_cut_pct": rel["carbon_cut_pct"],
                     "usd_saved": round(rel["energy_usd_saved"], 2)})
    return {
        "region_table": sustainability.region_table(),
        "jobs": rows,
        "total_carbon_saved_kg": round(total_saved_g / 1000, 1),
        "total_energy_usd_saved": round(total_usd_saved, 2),
        "cleanest": sustainability.cleanest_region(),
        "cheapest": sustainability.cheapest_region(),
    }


def run_all() -> dict:
    return {
        "ext1": ext1_tier_policy(),
        "ext2": ext2_rightsize_mbu(),
        "ext3": ext3_cache_economics(),
        "ext4": ext4_reasoning_budget(),
        "ext5": ext5_carbon_schedule(),
    }


def to_markdown(res: dict) -> str:
    e1, e2, e3, e4, e5 = res["ext1"], res["ext2"], res["ext3"], res["ext4"], res["ext5"]
    L = ["## Your Turn extensions (measured)", ""]

    L += ["### Ext 1 — smarter `recommend_tier` (spot churn + 1yr vs 3yr)", "",
          f"- v1 policy monthly: **${e1['v1_monthly']:,}** ({e1['v1_savings_pct']}% vs on-demand)",
          f"- v2 policy monthly: **${e1['v2_monthly']:,}** ({e1['v2_savings_pct']}% vs on-demand)",
          "",
          "| job | gpu | v1 | v2 | why v2 |", "|---|---|---|---|---|"]
    for r in e1["table"]:
        L.append(f"| {r['job_id']} | {r['gpu']} | {r['v1']} | {r['v2']} | {r['why']} |")
    L += ["",
          f"v2 lands at **{e1['v2_savings_pct']}%** vs v1's **{e1['v1_savings_pct']}%** — lower, and that is the point: "
          "every job in `workloads.csv` runs <365 days, so a 3-year lock-in is not justified (v2 picks 1yr at -20%), "
          "and the L4/A10G spot jobs churn at 12-15%/hr so v2 pulls them off spot. v2 trades headline savings for a "
          "commitment profile that survives contact with reality.", ""]

    L += ["### Ext 2 — right-size memory-bound GPUs by MBU (not $/GPU-hr)", "",
          "$/hr per TB/s of HBM bandwidth (the honest unit for decode):", "",
          "| GPU | $/(TB/s)·hr |", "|---|---|"]
    for g, v in sorted(e2["price_per_tbs"].items(), key=lambda x: x[1]):
        L.append(f"| {g} | {v} |")
    L += ["",
          f"Memory-bound cards flagged: `{e2['candidates']}`.", ""]
    if e2["picks"]:
        L += ["| GPU | from | to | achieved TB/s | $/hr saved |", "|---|---|---|---|---|"]
        for p in e2["picks"]:
            L.append(f"| {p['gpu_id']} | {p['from']} | {p['to']} | {p['achieved_tbs']} | {p['saved_per_hr']} |")
        L += ["", f"Right-sizing all of them saves **~${e2['monthly_saved']:,}/month**. "
              "You don't pick the cheapest $/GPU-hr card — you pick the cheapest card whose *bandwidth* still covers "
              "the workload's achieved demand with headroom, because decode never touches the extra FLOPs you'd be paying for.", ""]
    else:
        L += ["", "No cheaper card clears the 20% bandwidth-headroom bar for these workloads.", ""]

    be_gemini = pricing.cache_break_even_reads(write_cost_multiplier=3.0)
    L += ["### Ext 3 — `cache_is_worth_it()`", "",
          f"- Break-even with Anthropic-style pricing (write 1.25x, read 0.10x): **{e3['break_even_reads']} reads** "
          f"per prefix — i.e. caching almost always pays. With a storage-billed cache (Gemini-style, ~3x write): "
          f"**{be_gemini:.1f} reads**.",
          f"- Dataset: {e3['cached_requests']} cache-hit requests across ~{e3['distinct_prefixes_est']} shared prefixes "
          f"= **~{e3['avg_reads_per_prefix']} reads/prefix**.",
          f"- Verdict: `cache_is_worth_it` -> **{e3['worth_it']}** (and it correctly returns "
          f"{e3['worth_it_at_be_minus_1']} just below break-even). Caching is a landslide win here; the guard only "
          "matters for rarely-reused prefixes (one-off long documents).", ""]

    L += ["### Ext 4 — reasoning budget", "",
          f"- Reasoning is **{e4['reasoning_traffic_pct']}%** of traffic but "
          f"**{e4['reasoning_cost_pct']}%** of inference cost and **{e4['reasoning_energy_pct']}%** of energy.",
          f"- Reasoning requests: {e4['reasoning']['requests']} @ ${e4['reasoning']['cost']} / {e4['reasoning']['wh']:,} Wh; "
          f"normal: {e4['normal']['requests']} @ ${e4['normal']['cost']} / {e4['normal']['wh']:,} Wh.",
          f"- Per-request energy is **{e4['energy_multiple']}x** a normal query: a reasoning trace emits ~6x the output "
          "tokens *and* every token carries the ~80x reasoning energy multiplier (long autoregressive decode, no "
          "batching headroom, KV-cache growth).",
          f"- Proposal: gate reasoning on a task-complexity score; assume ~60% of current reasoning calls are "
          f"over-triggered. Re-routing them saves **${e4['gate_cost_saved']}/day** and "
          f"**{e4['gate_wh_saved']:,} Wh/day** (~{round(e4['gate_wh_saved']/1000*380)} gCO2e/day at us-east-1).", ""]

    L += ["### Ext 5 — carbon-aware scheduling", "",
          f"Cleanest region: **{e5['cleanest']}** (30 gCO2/kWh, hydro). Cheapest: **{e5['cheapest']}**.", "",
          "| region | $/kWh | gCO2/kWh | blended rank |", "|---|---|---|---|"]
    for r in e5["region_table"]:
        L.append(f"| {r['region']} | {r['usd_per_kwh']} | {r['gco2_per_kwh']} | {r['blended_score']} |")
    L += ["", "| job | gpu | kWh | carbon cut | $ elec saved |", "|---|---|---|---|---|"]
    for r in e5["jobs"]:
        L.append(f"| {r['job_id']} | {r['gpu']} | {r['kwh']:,} | {r['carbon_cut_pct']}% ({r['carbon_saved_kg']} kg) | ${r['usd_saved']} |")
    L += ["",
          f"Moving every interruptible job from `us-east-1` to `{e5['cleanest']}` cuts "
          f"**~{e5['total_carbon_saved_kg']:,} kg CO2e/month** and **${e5['total_energy_usd_saved']:,}/month** of "
          "electricity. Trade-off: `europe-north1` is far from most users, so this only applies to interruptible "
          "training/eval jobs that are latency-insensitive — never to the chat inference path.", ""]

    return "\n".join(L)


if __name__ == "__main__":
    res = run_all()
    md = to_markdown(res)
    out = os.path.join(ROOT, "outputs", "extensions.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\nWritten: outputs/extensions.md")
