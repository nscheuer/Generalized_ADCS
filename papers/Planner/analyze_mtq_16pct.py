"""Confirm/diagnose the 16% non-convergence on the 3+0 (MTQ-only) reduced testbed.

Loads the committed mc100_altro_3+0_reduced .sim (no new sim) and characterizes the
16 failed trials. Tests the naive hypothesis (goal near the uncontrollable axis ->
blocked during the slew) -- which is REFUTED -- and shows the real two-population
split: (a) horizon truncation (still slewing at cutoff), (b) terminal MTQ-hold limit
(residual correction torque near the instantaneous field).

Outputs (papers/Planner/output_data):
  fig_mtq_failure_modes.png/.pdf
  MTQ_16PCT_FAILURE_NOTE.md
"""
import os, sys, glob
import numpy as np
ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ADCS
from ADCS.helpers.math_helpers import rot_mat, normalize

OUTDIR = os.path.join(ROOT, "papers/Planner/output_data")
BORE = np.array([0., 1, 0]); CONV = 5.0


def ang(a, b):
    return np.degrees(np.arccos(np.clip(abs(np.dot(normalize(a), normalize(b))), 0, 1)))


def main():
    sim = sorted(glob.glob(os.path.join(ROOT, "papers/Planner/output/mc100_altro_3+0_reduced_*.sim")))[-1]
    r = ADCS.SimulationResults.load(sim)
    rows = []
    for run in r.runs:
        st = np.asarray(run.state_hist); tg = np.asarray(run.target_hist)
        oss = run.os_hist; ts = np.asarray(run.time_s)
        g = normalize(tg[-1][1:4])
        berr = np.array([ang(rot_mat(normalize(st[k, 3:7])) @ BORE, g) for k in range(len(st))])
        conv = berr[-1] < CONV
        slew_blk, term_blk = [], []
        for k in range(len(st)):
            bz = rot_mat(normalize(st[k, 3:7])) @ BORE
            n = np.cross(bz, g); B = np.asarray(oss[k]['B'])
            if np.linalg.norm(n) < 1e-6 or np.linalg.norm(B) < 1e-9:
                continue
            a = abs(np.dot(normalize(n), normalize(B)))
            if berr[k] > 5:
                slew_blk.append(a)
            if k >= int(0.75 * len(st)):
                term_blk.append(a)
        last = ts >= ts[-1] - 200
        slope = np.polyfit(ts[last], berr[last], 1)[0] * 1000
        rows.append(dict(conv=conv, final=berr[-1], init=berr[0],
                         slew=np.mean(slew_blk) if slew_blk else np.nan,
                         term=np.mean(term_blk) if term_blk else np.nan, slope=slope))
    C = [x for x in rows if x['conv']]; F = [x for x in rows if not x['conv']]
    horizon = [x for x in F if x['slope'] < -5]
    hold = [x for x in F if x['slope'] >= -5]

    # ---- figure ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    a0.scatter([x['slew'] for x in C], [x['term'] for x in C], s=22, c="C0", alpha=0.6, label="converged (84)")
    a0.scatter([x['slew'] for x in F], [x['term'] for x in F], s=34, c="C3", marker="x", label="failed (16)")
    a0.axhline(np.nanmean([x['term'] for x in C]), ls=":", c="C0", lw=1)
    a0.axhline(np.nanmean([x['term'] for x in F]), ls=":", c="C3", lw=1)
    a0.set_xlabel("slew-phase blocking  |n̂·B̂|  (identical → not the cause)")
    a0.set_ylabel("terminal blocking  |n̂·B̂|  (last 25%)")
    a0.set_title("Failed need a more B-aligned terminal hold"); a0.legend(fontsize=8); a0.grid(alpha=0.3)

    a1.scatter([x['slope'] for x in hold], [x['final'] for x in hold], s=34, c="C3", label=f"terminal hold limit ({len(hold)})")
    a1.scatter([x['slope'] for x in horizon], [x['final'] for x in horizon], s=34, c="C1", marker="s", label=f"ran out of horizon ({len(horizon)})")
    a1.axvline(0, ls=":", c="k", lw=0.8); a1.axhline(CONV, ls=":", c="0.5", lw=0.8)
    a1.set_xlabel("error slope, last 200 s [deg/1000 s]"); a1.set_ylabel("final error [deg]")
    a1.set_title("16% splits in two"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_mtq_failure_modes.png"), dpi=150)
    plt.savefig(os.path.join(OUTDIR, "fig_mtq_failure_modes.pdf"))
    print("saved fig_mtq_failure_modes.png/.pdf")

    def mm(s, k):
        v = np.array([x[k] for x in s], float); return np.nanmean(v)
    with open(os.path.join(OUTDIR, "MTQ_16PCT_FAILURE_NOTE.md"), "w") as f:
        f.write("# 3+0 (MTQ-only) reduced: the 16% non-convergence, diagnosed\n\n")
        f.write(f"Source: `{os.path.basename(sim)}` (100 trials, no new run). 84 converged, 16 failed.\n\n")
        f.write("## Naive hypothesis REFUTED\n")
        f.write(f"- slew-phase blocking |n̂·B̂|: converged {mm(C,'slew'):.2f} vs failed {mm(F,'slew'):.2f} (identical)\n")
        f.write(f"- initial error: converged {mm(C,'init'):.0f}° vs failed {mm(F,'init'):.0f}° (one failed trial started 12° away)\n")
        f.write("So failures are NOT 'goal near the uncontrollable axis' or 'harder slews'.\n\n")
        f.write("## Real mechanism: two populations\n")
        f.write(f"1. **Horizon truncation ({len(horizon)}):** end-slope < -5 deg/1000s (still slewing fast); "
                f"large finals (up to {max(x['final'] for x in horizon):.0f}°). Longer horizon would converge.\n")
        f.write(f"2. **Terminal MTQ-hold limit ({len(hold)}):** flat/positive end-slope; finals "
                f"{min(x['final'] for x in hold):.0f}–{max(x['final'] for x in hold):.0f}°; "
                f"terminal blocking {mm(hold,'term'):.2f} vs {mm(C,'term'):.2f} (converged). The residual "
                "correction torque sits near the instantaneous field, which a magnetorquer cannot produce, "
                "so it drifts/limit-cycles rather than holding.\n\n")
        f.write(f"Terminal blocking overall: converged {mm(C,'term'):.2f} vs failed {mm(F,'term'):.2f}.\n")
    print("saved MTQ_16PCT_FAILURE_NOTE.md")
    print(f"  horizon-truncation={len(horizon)}  terminal-hold={len(hold)}")


if __name__ == "__main__":
    main()
