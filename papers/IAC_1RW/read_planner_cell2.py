"""Cell-2 (1rw_full_planner) reads, per Addenda 4/4b/4c and the read-protection rule.

Order: headline BOTH WAYS (all trials / pure-planner with kill-affected excluded);
fallback split; kill-trial detail (per-window walls -- 4b's windows-2-3 discriminator);
solve-time histogram + median-vs-window-index (phase row); task-class tally.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import cell_metrics, error_series, T_ORBIT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")


def main():
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "A_trials", "1rw_full_planner_seed*.pkl"))):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    say(f"loaded {len(runs)} trials (1rw_full_planner)")
    assert len(runs) == 100, "cell incomplete -- do not adjudicate"

    kill_trials = [r for r in runs if r.get("n_budget_kills", 0)]
    clean = [r for r in runs if not r.get("n_budget_kills", 0)]
    kill_seeds = sorted(int(r["config"]["seed"]) for r in kill_trials)

    def headline(rs, label):
        m = cell_metrics(rs, T_ORBIT)
        say(f"{label}: n={len(rs)}  conv5 {m['conv_pct_5deg']:.1f}%  conv1 "
            f"{m['conv_pct_1deg']:.1f}%  median {m['median_final_deg']:.2f} deg  "
            f"fallback_frac {m['mean_fallback_frac']}  kills {m['total_budget_kills']} "
            f"solve-failures {m['total_solve_failures']}")
        return m

    say("\n== headline, both ways (read protection: no blended cell numbers) ==")
    m_all = headline(runs, "ALL TRIALS      ")
    headline(clean, "PURE-PLANNER    ")
    say(f"kill-affected trials, separately: {kill_seeds}")
    for r in kill_trials:
        e = error_series(r)
        say(f"  seed {int(r['config']['seed']):3d}: final {float(e[-1]):7.2f} deg, "
            f"kills {r['n_budget_kills']}, plans {r['n_plans']}, "
            f"fb_frac {r['n_fallbacks']/(r['n_plans']+r['n_fallbacks']):.2f}")

    say("\n== 4b: kill-trial window profiles (walls in s; K = killed ~300) ==")
    for r in kill_trials:
        pw = [f"{v:5.0f}" for v in r.get("plan_wall_s", [])]
        kt = [int(v // 500) for v in r.get("budget_kill_t", [])]
        say(f"  seed {int(r['config']['seed']):3d}: walls {' '.join(pw)}  "
            f"killed windows {kt}")
    say("  (4b separator: killed-trial windows 2-3 normal ~50 s => draw property; "
        "elevated => cascade)")

    say("\n== 4c phase row + Addendum 4 discriminator 2: solve-time structure ==")
    walls_by_widx = {}
    all_walls = []
    for r in runs:
        pw = r.get("plan_wall_s") or []
        for k, v in enumerate(pw):
            walls_by_widx.setdefault(k, []).append(float(v))
            all_walls.append(float(v))
    say("  median wall by window index (ALL trials):")
    say("  idx: " + " ".join(f"{k:5d}" for k in sorted(walls_by_widx)[:12]))
    say("  med: " + " ".join(f"{np.median(walls_by_widx[k]):5.1f}"
                             for k in sorted(walls_by_widx)[:12]))
    aw = np.asarray(all_walls)
    say(f"  histogram (n={len(aw)}): p50 {np.percentile(aw,50):.1f}  p90 "
        f"{np.percentile(aw,90):.1f}  p99 {np.percentile(aw,99):.1f}  "
        f">=290s (censored kills) {int((aw>=290).sum())}")
    say(f"  bimodality read: fraction in [2x median, 290s] = "
        f"{float(np.mean((aw > 2*np.median(aw)) & (aw < 290))):.4f} "
        f"(near-zero + a censored spike = bimodal wedge class; smooth tail = threshold artifact)")

    k = len(kill_trials)
    n_kills = int(m_all["total_budget_kills"])
    say(f"\n== clustering baseline: {n_kills} kills across ~{sum(r.get('n_plans',0)+r.get('n_fallbacks',0) for r in runs)} windows, "
        f"in {k} trials ==")
    say("  (independence would spread kills ~1/trial; compute exact binomial in the "
        "writeup only if the crossover leaves the draw branch alive)")

    say("\n== task-class tally ==")
    say(f"  reduced cell: 0 kills / 1200 windows. full cell: {n_kills} kills, "
        f"{k} trials, all in the late-dispatched cohort: "
        f"{all(s >= 75 for s in kill_seeds)}")

    with open(os.path.join(OUT, "PLANNER_CELL2_READS.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    say("\nwritten: output_data/PLANNER_CELL2_READS.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
