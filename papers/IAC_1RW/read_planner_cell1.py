"""Pre-registered reads for the planner money cell (1rw_reduced_planner, n=100).

Adjudicates, exactly as written and in this order:
  1. fallback fraction / split (is the cell a planner measurement?)
  2. RETUNE_PREDICTION preconditions (held-window along-B energy fraction + deviation median)
  3. CLAMP Addendum 3.1 screen validation (planner divergences vs frozen F)
  4. CLAMP Addendum 3.2 scheduling rescue (flagged seeds, paired planner-vs-PD)
  5. CLAMP Addendum 4 (W = budget-killed seeds vs F)
Writes output_data/PLANNER_CELL1_READS.txt
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import cell_metrics, error_series, T_ORBIT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")
F = [8, 12, 16, 23, 29, 49, 55, 78, 85]          # frozen 2026-08-19, pre-planner-data
HELD_T0 = 1000.0                                  # held window: post-acquisition
DIVERGE_DEG = 30.0                                # PD-half bimodality gap criterion


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "A_trials", "1rw_reduced_planner_seed*.pkl"))):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    say(f"loaded {len(runs)} planner trials (1rw_reduced)")
    assert len(runs) == 100, "cell incomplete -- do not adjudicate"

    m = cell_metrics(runs, T_ORBIT)
    say(f"\n== headline (t = {T_ORBIT:.0f} s) ==")
    say(f"conv@5deg {m['conv_pct_5deg']:.1f}%  conv@1deg {m['conv_pct_1deg']:.1f}%  "
        f"median {m['median_final_deg']:.2f} deg  held-p95 {m['median_held_p95_deg']:.2f} deg  "
        f"KNOWLEDGE {m.get('median_est_att_err_deg', float('nan')):.3f} deg")

    # ---- READ 1: fallback split ----
    say("\n== READ 1: fallback fraction / split ==")
    say(f"mean_fallback_frac = {m['mean_fallback_frac']}")
    say(f"total plans = {sum(r.get('n_plans', 0) for r in runs)}, "
        f"total fallbacks = {sum(r.get('n_fallbacks', 0) for r in runs)} "
        f"(budget-kills {m['total_budget_kills']}, solve-failures {m['total_solve_failures']}, "
        f"track {m['total_track_fallbacks']})")
    say("VERDICT: cell IS a planner measurement" if (m["mean_fallback_frac"] or 0) < 0.2
        else "VERDICT: fallback-dominated -- NOT a planner measurement")

    # ---- per-seed finals ----
    finals = {}
    for r in runs:
        e = error_series(r)
        finals[int(r["config"]["seed"])] = float(e[-1])
    diverged = sorted(s for s, v in finals.items() if v > DIVERGE_DEG)

    # ---- READ 2: RETUNE preconditions (held-window, n=100) ----
    say("\n== READ 2: RETUNE_PREDICTION preconditions (held window t >= "
        f"{HELD_T0:.0f} s) ==")
    fr_list, dev_list = [], []
    for r in runs:
        t = np.asarray(r["time"], float)
        held = t >= HELD_T0
        dv = np.asarray(r["plan_deviation"], float)
        da = np.asarray(r["plan_dev_alongB"], float)
        dp = np.asarray(r["plan_dev_perpB"], float)
        ok = held & np.isfinite(dv) & (dv >= 0)
        if ok.sum() < 10:
            continue
        dev_list.append(float(np.median(dv[ok])))
        e_along = float(np.sum(da[ok] ** 2))
        e_perp = float(np.sum(dp[ok] ** 2))
        if e_along + e_perp > 0:
            fr_list.append(e_along / (e_along + e_perp))
    say(f"trials with held-window plan data: {len(dev_list)}")
    say(f"held-window deviation median (across-trial median): {np.median(dev_list):.3f} deg")
    say(f"held-window along-B energy fraction (median): {np.median(fr_list):.3f} "
        f"(isotropy 1/3 = 0.333; smoke read 0.451)")
    pre1 = np.median(fr_list) > 0.40
    pre2 = 0.5 <= np.median(dev_list) <= 2.0
    pre3 = (m["mean_fallback_frac"] or 0) < 0.2
    say(f"preconditions: along-B > 0.40: {pre1}; deviation ~1 deg [0.5, 2.0]: {pre2}; "
        f"low fallback: {pre3}")
    say("VERDICT: retune rerun " + ("JUSTIFIED -- invoke the 2-cell retune"
        if (pre1 and pre2 and pre3) else "NOT justified -- frozen-half numbers stand"))

    # ---- READ 3: screen validation ----
    say("\n== READ 3: Addendum 3.1 screen validation ==")
    say(f"planner diverged seeds (> {DIVERGE_DEG:.0f} deg): {diverged}")
    say(f"frozen flagged set F: {F}")
    inside = [s for s in diverged if s in F]
    say(f"inside F: {inside}  ({len(inside)}/{len(diverged) if diverged else 0})")

    # ---- READ 4: scheduling rescue (paired seeds vs the documented PD diverged set) ----
    say("\n== READ 4: Addendum 3.2 scheduling rescue (paired seeds) ==")
    # PD money cell (clamped, n=100): 11 diverged, all pinned at h = 1.0000 --
    # identities documented at adjudication time: the 9 flagged + outliers 15, 53.
    PD_DIVERGED = sorted(F + [15, 53])
    say(f"PD diverged set (documented): {PD_DIVERGED}")
    say("planner finals on those seeds:")
    resc = []
    for s in PD_DIVERGED:
        b = finals.get(s)
        tag = ""
        if b is not None and b < 5.0:
            tag = "  <- RESCUED (PD diverged -> planner converged)"
            resc.append(s)
        elif b is not None and b > DIVERGE_DEG:
            tag = "  <- still diverged"
        say(f"  seed {s:3d}: planner final {b:8.2f} deg{tag}")
    say(f"rescued: {len(resc)}/11 -> {resc}")
    say(f"outliers 15/53 (registered as second mechanism, predicted NOT rescued by "
        f"scheduling): 15 -> {finals.get(15, float('nan')):.2f}, "
        f"53 -> {finals.get(53, float('nan')):.2f}")
    nov = [s for s in diverged if s not in PD_DIVERGED]
    if nov:
        say(f"NOVEL planner-only divergences (converged under PD): {nov}")
        for s in nov:
            r = next(x for x in runs if int(x["config"]["seed"]) == s)
            hf = np.asarray(r["h_frac"], float)
            sg = np.asarray(r["sigma"], float)
            say(f"  seed {s}: final {finals[s]:.1f} deg, h_frac end {hf[-1]:.3f} "
                f"(max {hf.max():.3f}), sigma median {np.median(sg[np.isfinite(sg)]):.3f}, "
                f"n_plans {r.get('n_plans')}, dwell(sigma<0.2) "
                f"{float(np.mean(sg[np.isfinite(sg)] < 0.2)):.4f} (screen flags <= 0.1035)")

    # ---- READ 5: Addendum 4 ----
    say("\n== READ 5: Addendum 4 (solver-hostile seeds vs F) ==")
    W = m["budget_kill_seeds"]
    say(f"W (budget-killed seeds, primary cell) = {W}")
    if not W:
        say("W is EMPTY: zero budget kills in 100/100 trials under the 300 s hard budget.")
        say("The two 14.5 h wedges of the pre-hardening run DID NOT REPRODUCE on the same")
        say("seeds/configs => the hang is a STOCHASTIC SOLVER EVENT, not a draw property.")
        say("No registered branch fires (table assumed |W| >= 1); adjudication: the overlap")
        say("question is MOOT; report the non-reproduction. Timing histogram (discriminator")
        say("2) comes from cell 2, which runs fully instrumented.")

    with open(os.path.join(OUT, "PLANNER_CELL1_READS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    say("\nwritten: output_data/PLANNER_CELL1_READS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
