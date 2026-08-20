"""Diagnose the ~1 deg planner floor: cost-weight equilibrium (a-terminal),
receding-horizon creep (c), or twin bias (b -- pre-excluded by kinematics).

All from existing cell-1 pkls: (1) finals histogram (tight cluster vs spread);
(2) per-trial kinematic identity |final - plan_err_end| <= dev_end; (3) late-orbit
error-profile slope (still decaying => creep; flat => equilibrium); (4) window-joint
sawtooth check (drops at replans => plans close early parts fast).
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import error_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data")


def main():
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    runs = []
    for p in sorted(glob.glob(os.path.join(OUT, "A_trials", "1rw_reduced_planner_seed*.pkl"))):
        with open(p, "rb") as f:
            runs.append(pickle.load(f))

    finals, devs_end, profs = [], [], []
    for r in runs:
        e = error_series(r)
        t = np.asarray(r["time"], float)
        dv = np.asarray(r["plan_deviation"], float)
        ok = np.isfinite(dv)
        finals.append(float(e[-1]))
        devs_end.append(float(dv[ok][-1]) if ok.any() else np.nan)
        profs.append((t, np.asarray(e, float)))
    finals = np.asarray(finals)
    conv = finals < 30.0

    say("== (1) finals histogram, converged trials ==")
    f = finals[conv]
    edges = np.arange(0.0, 3.01, 0.25)
    for a, b in zip(edges[:-1], edges[1:]):
        n = int(((f >= a) & (f < b)).sum())
        say(f"  [{a:4.2f},{b:4.2f}): {'#' * n}{n:3d}")
    say(f"  >=3.00: {int((f >= 3).sum())}")
    say(f"  median {np.median(f):.3f}  IQR [{np.percentile(f,25):.3f}, "
        f"{np.percentile(f,75):.3f}]  in [0.8,1.2): {int(((f>=0.8)&(f<1.2)).sum())}/99")

    say("\n== (2) kinematic identity: plan sits where execution sits ==")
    d = np.asarray(devs_end)[conv]
    say(f"  deviation at orbit end: median {np.nanmedian(d):.4f} deg, "
        f"p95 {np.nanpercentile(d,95):.4f} deg")
    say(f"  => plan-frame terminal error = executed final +/- dev_end; with median final "
        f"{np.median(f):.2f} deg vs dev {np.nanmedian(d):.4f} deg, the PLAN ITSELF sits "
        f"~{np.median(f):.2f} deg from goal. Twin bias (b) excluded.")

    say("\n== (3) late-orbit profile: creep vs flat ==")
    marks = [3500, 4000, 4500, 5000, 5400, 5553]
    med_at = []
    for tm in marks:
        vals = []
        for (t, e), c in zip(profs, conv):
            if not c:
                continue
            i = int(np.searchsorted(t, tm))
            if i < len(e):
                vals.append(e[i])
        med_at.append(float(np.median(vals)))
    say("  t(s):     " + "  ".join(f"{m:6d}" for m in marks))
    say("  med(deg): " + "  ".join(f"{v:6.3f}" for v in med_at))
    drop_late = med_at[0] - med_at[-1]
    say(f"  drop over last ~2000 s: {drop_late:.3f} deg "
        f"({'still DECAYING -> creep/(c)' if drop_late > 0.15 else 'FLAT -> equilibrium/(a-terminal)'})")

    say("\n== (4) window-joint sawtooth (median profile in replan-relative time) ==")
    # median error at offsets within the 500 s window, pooled over windows 8-11
    offs = np.arange(0, 500, 50)
    for w0 in (4000, 4500, 5000):
        row = []
        for off in offs:
            vals = []
            for (t, e), c in zip(profs, conv):
                if not c:
                    continue
                i = int(np.searchsorted(t, w0 + off))
                if i < len(e):
                    vals.append(e[i])
            row.append(float(np.median(vals)))
        say(f"  window t0={w0}: " + " ".join(f"{v:5.2f}" for v in row))
    say("  (monotone within-window decay + reset at joints would look sawtooth; "
        "smooth monotone across joints = plain slow convergence)")

    with open(os.path.join(OUT, "DIAG_1DEG.txt"), "w") as fo:
        fo.write("\n".join(lines) + "\n")
    say("\nwritten: output_data/DIAG_1DEG.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
