"""Figure (envelope slot): EMPIRICAL PERFORMANCE MAP -- where the tested cells
landed in (demand index D, converged-only median pointing error). No claimed
feasible region: D is a demand index, not a boundary.

Marker shape = controller (PD circle, planner triangle); colour = architecture
(house grammar); fill = outcome (filled if <= 2% large-error failures, open
otherwise with the divergence percentage beside it). Faint D = 1 line only.
"""
import glob
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import error_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style  # noqa: E402
fig_style.apply(10.0)
from fig_style import ARCH, CTRL  # noqa: E402

OUT = os.path.join(HERE, "output_data")
H_MAX = 15e-3


def cell_stats(pkls):
    fin, dend = [], []
    for p in pkls:
        with open(p, "rb") as f:
            r = pickle.load(f)
        fin.append(float(error_series(r)[-1]))
        dend.append(float(np.asarray(r["h_frac"], float)[-1]))
    fin = np.asarray(fin)
    conv = fin <= 30.0
    return dict(D=float(np.median(dend)), med=float(np.median(fin[conv])),
                div=float(100 * np.mean(~conv)))


def main():
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260818_202627.json")))
    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    D_ref = {r["alt_km"]: r for r in fj["rows_by_case"]["m_res=0.05"]}[400.0][
        "accum_along_wheel_Nms"] / H_MAX

    def jcell(k, wheel):
        h = d18["cells"][k]["horizons"]["5554"]
        div = 100.0 - h["conv_pct_5deg"]
        D = (float(np.median(np.asarray(h["per_trial_h_frac_end"], float)))
             if wheel else D_ref)          # 3+0: the demand index of the same bus/orbit
        return dict(D=D, med=float(h["median_final_deg"]), div=max(0.0, div))

    cells = {
        "reduced": [
            ("3+0", "PD", jcell("0rw_reduced_pd", False)),
            ("3+1", "PD", cell_stats(glob.glob(os.path.join(OUT, "wave/pd_reduced_kp1/*.pkl")))),
            ("3+1", "planner", cell_stats(glob.glob(os.path.join(OUT, "A_trials/1rw_reduced_planner_seed*.pkl")))),
            ("3+3", "PD", jcell("3rw_reduced_pd", True)),
        ],
        "full": [
            ("3+0", "PD", jcell("0rw_full_pd", False)),
            ("3+1", "PD", cell_stats(glob.glob(os.path.join(OUT, "wave/pd_full_kp1/*.pkl")))),
            ("3+1", "planner", cell_stats(glob.glob(os.path.join(OUT, "tune_seed*_wave_planner_full.pkl")))),
            ("3+3", "PD", jcell("3rw_full_pd", True)),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, sharex=True,
                             constrained_layout=True)
    for ax, (task, title) in zip(axes, (("reduced", "(a) Boresight pointing"),
                                        ("full", "(b) Full three-axis attitude"))):
        for thr in (1.0, 5.0, 30.0):
            ax.axhline(thr, color="#999999", lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=1)
            if ax is axes[-1]:
                ax.text(2.0, thr, f" {thr:.0f}°", fontsize=10, color="#777777",
                        ha="left", va="center", clip_on=False)
        ax.axvline(1.0, color="#999999", lw=0.8, ls="-", alpha=0.7)
        ax.text(1.0, 2.6e-2, "once-per-orbit\ndumping", fontsize=10, color="#777777",
                ha="center", va="bottom")
        for arch, ctrl, c in cells[task]:
            filled = c["div"] <= 2.0
            col = ARCH[arch]
            ax.plot(c["D"], c["med"], CTRL[ctrl]["marker"], ms=9.5,
                    mfc=col if filled else "white", mec=col, mew=1.6, zorder=3)
            if not filled:
                ax.annotate(f"{c['div']:.0f}%", (c["D"], c["med"]),
                            textcoords="offset points", xytext=(9, -3), fontsize=9.5,
                            color=col, va="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(2e-2, 2.0); ax.set_ylim(2e-2, 3e2)
        fig_style.log_decades_only(ax)
        ax.grid(True, which="major", axis="y")
        ax.set_title(title, loc="left")
        ax.set_xlabel("demand index $D$")
    axes[0].set_ylabel("achieved terminal error [deg]  (tighter $\\downarrow$)")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARCH[a], marker="s", ls="", ms=8, label=a)
               for a in ("3+0", "3+1", "3+3")]
    handles += [Line2D([], [], color="#444444", marker="o", ls="", ms=8, mfc="white",
                       mew=1.4, label="PD"),
                Line2D([], [], color="#444444", marker="^", ls="", ms=8, mfc="white",
                       mew=1.4, label="planner")]
    axes[1].legend(handles=handles, loc="upper right", ncol=1, handletextpad=0.4,
                   labelspacing=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_envelope.{ext}"))
    for task in cells:
        for arch, ctrl, c in cells[task]:
            print(f"{task:8s} {arch} {ctrl:8s} D={c['D']:.3f} med={c['med']:.2f} div={c['div']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
