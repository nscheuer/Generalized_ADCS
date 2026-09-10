#!/usr/bin/env python
"""Task 6 -- the missing "It Works" planner-vs-PD headline figure (Paper 2).

Paper 2 leads with planner-vs-PD. The 2x2x2 matrix already exists as overlaid
histograms (fig_p2_matrix), but the headline claim -- "the planner converges
where the PD baseline does not, on the controllability-appropriate task" -- reads
more cleanly as grouped bars. This builds that figure straight from the locked
stats JSON (papers/Planner/output_data/p2p1_matrix_stats.json), so it does NOT
re-run any MC or reload the .sim files.

Two panels, four config x goal cells, planner vs PD side by side:
  (left)  convergence rate  [% of 100 MC trials with final pointing error < 5 deg]
  (right) mean final pointing error [deg, log axis -- spans 0.26 to 85 deg]

The two controllability-appropriate cells (3+0/reduced, 3+1/full) are the diagonal
and are marked; the other two are the easy case (3+1/reduced, both succeed) and the
over-asked limit case (3+0/full, full 3-DOF on a 2-DOF-controllable plant -- both
fail, planner less badly).

Outputs (papers/Planner/output_data):
  fig_itworks_planner_vs_pd.png/.pdf
  ITWORKS_RESULTS.md
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
from papers._pubstyle import use_pub_style, DBL_W, PALETTE

OUTDIR = os.path.join(ROOT, "papers/Planner/output_data")
STATS = os.path.join(OUTDIR, "p2p1_matrix_stats.json")

# cell order: diagonal (controllability-appropriate) cells first so they read as
# the headline; easy + limit case after.
CELLS = [
    ("3+0", "reduced", "MTQ-only\nreduced goal", True),
    ("3+1", "full",    "3+1\nfull goal",          True),
    ("3+1", "reduced", "3+1\nreduced goal",       False),
    ("3+0", "full",    "MTQ-only\nfull goal",     False),
]


def main():
    use_pub_style()
    with open(STATS) as f:
        cells = json.load(f)["cells"]

    def get(cfg, goal, role):
        return cells[f"{cfg}|{goal}|{role}"]

    labels = [c[2] for c in CELLS]
    diag = np.array([c[3] for c in CELLS])
    pl_conv = np.array([get(c[0], c[1], "planner")["converged_pct"] for c in CELLS])
    pd_conv = np.array([get(c[0], c[1], "pd")["converged_pct"] for c in CELLS])
    pl_mean = np.array([get(c[0], c[1], "planner")["mean_deg"] for c in CELLS])
    pd_mean = np.array([get(c[0], c[1], "pd")["mean_deg"] for c in CELLS])
    pd_ctrl = [get(c[0], c[1], "pd")["controller"] for c in CELLS]

    x = np.arange(len(CELLS)); w = 0.38
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 3.1))

    # ---- left: convergence rate ----
    bL1 = axL.bar(x - w/2, pl_conv, w, color=PALETTE["planner"], label="Planner (ALTRO)")
    bL2 = axL.bar(x + w/2, pd_conv, w, color=PALETTE["pd"], label="PD baseline (Lovera / LP)")
    axL.set_ylabel("Converged  [%  $<5^\\circ$ final]")
    axL.set_ylim(0, 108)
    axL.set_title("Convergence rate (100-trial MC)")
    for b, v in list(zip(bL1, pl_conv)) + list(zip(bL2, pd_conv)):
        axL.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.0f}", ha="center",
                 va="bottom", fontsize=7)

    # ---- right: mean final error (log) ----
    bR1 = axR.bar(x - w/2, pl_mean, w, color=PALETTE["planner"], label="Planner (ALTRO)")
    bR2 = axR.bar(x + w/2, pd_mean, w, color=PALETTE["pd"], label="PD baseline")
    axR.set_ylabel("Mean final pointing error  [deg]")
    axR.set_ylim(0, 92)                    # linear (B3): planner bars near-zero vs tall PD bars
    axR.axhline(5.0, ls="--", lw=0.9, color=PALETTE["neutral"])
    axR.text(len(CELLS) - 0.5, 7.0, "$5^\\circ$ threshold", ha="right", va="bottom",
             fontsize=6.5, color=PALETTE["neutral"])
    axR.set_title("Mean final error (linear scale)")
    for b, v in list(zip(bR1, pl_mean)) + list(zip(bR2, pd_mean)):
        axR.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.2g}", ha="center",
                 va="bottom", fontsize=7)

    for ax in (axL, axR):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.grid(axis="x", visible=False)
        # mark the two controllability-appropriate (diagonal) cells
        for xi, isd in zip(x, diag):
            if isd:
                ax.get_xticklabels()[xi].set_fontweight("bold")

    # single shared legend up top (avoids colliding with the tall bars)
    fig.legend(handles=[bL1, bL2], labels=["Planner (ALTRO)", "PD baseline (Lovera / LP)"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94), fontsize=8)
    fig.suptitle("Planner vs PD baseline — converges where PD does not "
                 "(bold = controllability-appropriate task)", fontsize=9, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"fig_itworks_planner_vs_pd.{ext}"))
    plt.close(fig)
    print("saved fig_itworks_planner_vs_pd.png/.pdf")

    # ---- RESULTS.md ----
    with open(os.path.join(OUTDIR, "ITWORKS_RESULTS.md"), "w") as f:
        f.write("# Task 6 -- 'It Works': planner vs PD headline figure\n\n")
        f.write("Grouped-bar restatement of the 2x2x2 matrix (`fig_p2_matrix` histograms), "
                "built from `p2p1_matrix_stats.json` (100-trial MC, 1000 s, final error < 5 deg).\n\n")
        f.write("| Config / goal | role | controller | converged % | mean final err | median | p95 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for cfg, goal, lab, isd in CELLS:
            for role in ("planner", "pd"):
                c = get(cfg, goal, role)
                tag = " (diag)" if isd and role == "planner" else ""
                f.write(f"| {cfg} / {goal}{tag} | {role} | {c['controller']} | "
                        f"{c['converged_pct']:.0f}% | {c['mean_deg']:.2f}° | "
                        f"{c['median_deg']:.2f}° | {c['p95_deg']:.2f}° |\n")
        f.write("\n**Headline:** on the controllability-appropriate diagonal the planner converges "
                "where the PD baseline does not — MTQ-only reduced pointing 84% vs 27% (mean 3.85° vs "
                "21.5°), and 3+1 full attitude 94% vs 90% but mean 1.19° vs 9.06° (the PD tail is much "
                "heavier: p95 10.1° vs 74.8°). On the easy case (3+1 reduced) both succeed (99/97%). On "
                "the over-asked limit case (3+0 full = full 3-DOF on a 2-DOF-controllable plant) both "
                "fail, but the planner degrades less (18% vs 0%, mean 18.3° vs 85.2°).\n")
    print("saved ITWORKS_RESULTS.md")


if __name__ == "__main__":
    main()
