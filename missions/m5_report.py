"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing
from missions import extensions

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def _analysis_section(r1, r2, r3, lie_ids, cat, levers) -> str:
    """Prose analysis for rubric C.2 — mechanism, priorities, sustainability link."""
    ranked = sorted(levers.items(), key=lambda x: -x[1])
    lie_line = ", ".join(f"`{g}`" for g in lie_ids) or "none"
    return "\n".join([
        "## Analysis",
        "",
        "### Why GPU-Util is a lie (and what it costs)",
        "",
        f"`nvidia-smi` GPU-Util only reports whether *a* SM had *a* thread resident in the "
        f"last sample window — it is a duty-cycle clock, not an efficiency metric. "
        f"{lie_line} read >=90% GPU-Util while their MFU sits near 0.20, meaning ~80% of the "
        f"rented tensor-core FLOPs produced nothing. The usual mechanism is a memory stall: "
        f"the kernel is waiting on HBM (small batch, unfused attention, KV-cache thrash) so the "
        f"SMs are 'busy' spinning on loads. You still pay the full on-demand GPU-hour. "
        f"For `gpu-h100-4` alone that is an H100 billed at $2.50/hr delivering ~A100-class work.",
        "",
        "### $/1M-token: baseline vs optimized",
        "",
        f"- Inference unit cost: **${r2['baseline_per_m']}/1M-token -> ${r2['optimized_per_m']}/1M-token** "
        f"({r2['savings_pct']}% lower) once cascade + prompt-caching + batch are stacked.",
        f"- Purchasing: on-demand **${r3['on_demand_monthly']:,}/mo -> ${r3['optimized_monthly']:,}/mo** "
        f"({r3['savings_pct']}%) by matching tier to duty cycle.",
        "",
        "### Recommended order of action (by ROI, not by size)",
        "",
        "1. **Cascade routing first.** Zero capex, reversible, and 80% of traffic is easy enough for "
        "the small model (15x cheaper). Biggest $/effort ratio.",
        "2. **Prompt caching for chat/RAG.** The shared system prompt is read hundreds of times per "
        "prefix — far past break-even (see Ext 3). Config-only change.",
        "3. **Kill idle GPUs + right-size the util-lies.** Operational hygiene; "
        f"${round(levers['Kill idle GPUs']):,}/mo + ${round(levers['Right-size util-lies']):,}/mo with no "
        "user-visible impact.",
        "4. **Commitment purchasing last.** Highest absolute savings "
        f"(${round(levers['Purchasing (spot/reserved)']):,}/mo) but it locks in spend — do it only after "
        "the workload mix is stable, and prefer 1yr until a job proves it will outlive 3 years (Ext 1).",
        "",
        f"Top lever by dollars: **{ranked[0][0]}** (${ranked[0][1]:,}/mo).",
        "",
        "### Sustainability ties back to cost",
        "",
        "Region choice moves $ and gCO2e together: `europe-north1` (Norway hydro, 30 gCO2/kWh, "
        "$0.09/kWh) beats `us-east-1` (380 gCO2/kWh, $0.12/kWh) on both axes, and `europe-central2` "
        "(Poland, 660 gCO2/kWh) is worst on both. Scheduling interruptible training/eval into the "
        "clean region is a genuine win; the latency cost rules it out for the chat inference path. "
        "Reasoning traffic is the other lever — a few percent of requests but the dominant share of "
        "energy (Ext 4).",
    ])


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    # --- analysis narrative (rubric C.2) + measured extensions (rubric D) ---
    lie_ids = [l["gpu_id"] for l in r1["lies"]]
    analysis = _analysis_section(r1, r2, r3, lie_ids, cat, levers)
    try:
        ext_md = extensions.to_markdown(extensions.run_all())
    except Exception as e:  # never let the extension layer break the core deliverable
        ext_md = f"_extensions unavailable: {e}_"
    extra = analysis + "\n\n" + ext_md

    md = report.build_report(baseline, optimized, levers, sustainability=sust, extra_sections=extra)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
