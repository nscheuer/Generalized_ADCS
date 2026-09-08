"""Screening diagram, candidate v2: the grid cells of fig3_envelope plus the
analytic structure the paper derives and the closed-loop variant cells it
discusses but never plots.

Vertical (D axis):  D = 1 (one wheel capacity per orbit); the per-orbit dump
capacity band at tau_allow = 2.5 uN m (nadir 1.57 .. inertial 2.42 h_max), the
momentum boundary expressed in D; a top axis giving the wheel capacity that
puts the reference bus at each D (D ~ 1/h_max).
Horizontal (error axis): PD settling floor tau_res/k_p (drag+GG .. +dipole
residual), knowledge-error band, and the 1/5/30 deg thresholds.
Variants (small open markers): QP allocator on 3+0; doubled k_p, untuned
planner weights, and 15 deg inclination on 3+1.
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
KP = 2.9e-4
GREY = "#777777"


def cell_stats(pkls):
    fin, dend = [], []
    for p in pkls:
        with open(p, "rb") as f:
            r = pickle.load(f)
        fin.append(float(error_series(r)[-1]))
        dend.append(float(np.asarray(r["h_frac"], float)[-1]))
    fin = np.asarray(fin)
    return dict(D=float(np.median(dend)), med=float(np.median(fin)),
                div=float(100 * np.mean(fin > 30.0)))


def main():
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260907_184702.json")))
    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    prof = fj["profile_comparison_400km"]
    D_ref = prof["nadir"]["accum_along_wheel_Nms"] / H_MAX
    cap_nadir = prof["nadir"]["capacity_per_orbit_Nms"] / H_MAX
    cap_inert = prof["inertial"]["capacity_per_orbit_Nms"] / H_MAX

    def jcell(k, wheel):
        h = d18["cells"][k]["horizons"]["5554"]
        fin = np.asarray(h["finals_deg"], float)
        D = (float(np.median(np.asarray(h["per_trial_h_frac_end"], float))) if wheel else D_ref)
        return dict(D=D, med=float(np.median(fin)), div=float(100 * np.mean(fin > 30.0)))

    g = lambda p: glob.glob(os.path.join(OUT, p))  # noqa: E731
    cells = {
        "reduced": [("3+0", "PD", jcell("0rw_reduced_pd", False)),
                    ("3+1", "PD", cell_stats(g("wave/pd_reduced_kp1/*.pkl"))),
                    ("3+1", "planner", cell_stats(g("A_trials/1rw_reduced_planner_seed*.pkl"))),
                    ("3+3", "PD", jcell("3rw_reduced_pd", True))],
        "full": [("3+0", "PD", jcell("0rw_full_pd", False)),
                 ("3+1", "PD", cell_stats(g("wave/pd_full_kp1/*.pkl"))),
                 ("3+1", "planner", cell_stats(g("tune_seed*_wave_planner_full.pkl"))),
                 ("3+3", "PD", jcell("3rw_full_pd", True))],
    }
    qp = cell_stats(g("wave/qp_0rw_reduced/*.pkl")); qp["D"] = D_ref
    variants = {
        "reduced": [("3+0", "PD", "QP allocator", qp, (8, 0), "left"),
                    ("3+1", "PD", "$i=15^\\circ$", cell_stats(g("lowinc/*.pkl")), (8, 0), "left")],
        "full": [("3+1", "PD", "$2k_p$", cell_stats(g("wave/pd_full_kp2/*.pkl")), (0, -12), "center"),
                 ("3+1", "planner", "untuned weights", cell_stats(g("tune_seed*_wave_planner_full_base.pkl")), (8, 0), "left")],
    }
    # PD settling floor tau_res / k_p: drag+GG (0.42 uN m) to +24% dipole residual (~0.8 uN m)
    floor_lo, floor_hi = np.degrees(0.42e-6 / KP), np.degrees(0.8e-6 / KP)
    know_lo, know_hi = 0.004, 0.019

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.3), sharey=True, sharex=True,
                             constrained_layout=True)
    for ax, (task, title) in zip(axes, (("reduced", "(a) Boresight pointing"),
                                        ("full", "(b) Full three-axis attitude"))):
        # horizontal structure
        for thr in (1.0, 5.0, 30.0):
            ax.axhline(thr, color="#999999", lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=1)
            if ax is axes[-1]:
                ax.text(4.0, thr, f" {thr:.0f}°", fontsize=10, color=GREY, ha="left",
                        va="center", clip_on=False)
        ax.axhspan(floor_lo, floor_hi, color="#000000", alpha=0.06, lw=0, zorder=0)
        ax.axhspan(know_lo, know_hi, color="#000000", alpha=0.06, lw=0, zorder=0)
        ax.text(0.85, np.sqrt(floor_lo * floor_hi), "PD floor $\\tau_{\\rm res}/k_p$",
                fontsize=8.5, color=GREY, ha="right", va="center")
        ax.text(0.85, np.sqrt(know_lo * know_hi), "knowledge error", fontsize=8.5, color=GREY,
                ha="right", va="center")
        # vertical structure
        ax.axvline(1.0, color="#999999", lw=0.8, alpha=0.7, zorder=1)
        ax.axvspan(cap_nadir, cap_inert, color="#E69F00", alpha=0.15, lw=0, zorder=0)
        ax.text(0.93, 12.0, "one wheel capacity per orbit", fontsize=8, color=GREY,
                ha="right", va="center", rotation=90, clip_on=True)
        ax.text(np.sqrt(cap_nadir * cap_inert), 12.0, "dump capacity per orbit",
                fontsize=8, color="#9A5B00", ha="center", va="center", rotation=90)
        # variant cells first (behind), then the grid cells
        for arch, ctrl, label, c, off, ha in variants[task]:
            col = ARCH[arch]
            ax.plot(c["D"], c["med"], CTRL[ctrl]["marker"], ms=6.5, mfc="white", mec=col,
                    mew=1.2, zorder=3)
            ax.annotate(f"{label} ({c['div']:.0f}%)", (c["D"], c["med"]), textcoords="offset points",
                        xytext=off, fontsize=8.5, color=col, va="center", ha=ha)
        for arch, ctrl, c in cells[task]:
            filled = c["div"] <= 2.0
            col = ARCH[arch]
            ax.plot(c["D"], c["med"], CTRL[ctrl]["marker"], ms=9.5,
                    mfc=col if filled else "white", mec=col, mew=1.6, zorder=4)
            if True:
                left = (task == "reduced" and arch == "3+1" and ctrl == "PD")
                ax.annotate(f"{c['div']:.0f}%", (c["D"], c["med"]), textcoords="offset points",
                            xytext=(-9, 0) if left else (9, -3), fontsize=9.5, color=col,
                            va="center", ha="right" if left else "left")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(2e-2, 4.0); ax.set_ylim(2e-3, 3e2)
        fig_style.log_decades_only(ax)
        ax.grid(True, which="major", axis="y")
        ax.text(0.02, 0.985, title, transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.set_xlabel("demand index $D$")
        # top axis: wheel capacity that places the reference bus at D (D ~ 1/h_max)
        top = ax.twiny(); top.set_xscale("log"); top.set_xlim(ax.get_xlim())
        hs = [50, 15, 5, 1.5, 0.5]
        top.set_xticks([D_ref * 15.0 / h for h in hs]); top.set_xticklabels([str(h) for h in hs])
        top.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        top.tick_params(labelsize=9, colors=GREY, length=3)
        top.spines["top"].set_visible(True); top.spines["top"].set_color("#BBBBBB")
        top.spines["right"].set_visible(False)
        top.set_xlabel("equivalent wheel capacity for the reference disturbance [mN m s]", fontsize=9,
                       color=GREY, labelpad=4)
    axes[0].set_ylabel("achieved terminal error [deg]  (tighter $\\downarrow$)")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ARCH[a], marker="s", ls="", ms=8, label=a) for a in ("3+0", "3+1", "3+3")]
    handles += [Line2D([], [], color="#444444", marker="o", ls="", ms=8, mfc="white", mew=1.4, label="PD"),
                Line2D([], [], color="#444444", marker="^", ls="", ms=8, mfc="white", mew=1.4, label="planner"),
                Line2D([], [], color="#444444", marker="o", ls="", ms=5.5, mfc="white", mew=1.0, label="variant")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, handletextpad=0.4,
               columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_envelope_v2.{ext}"))
    print(f"D_ref {D_ref:.3f}; dump capacity band {cap_nadir:.2f}-{cap_inert:.2f}; floor {floor_lo:.3f}-{floor_hi:.3f} deg")
    for task in variants:
        for arch, ctrl, label, c, *_ in variants[task]:
            print(f"{task:8s} variant {label:16s} D={c['D']:.3f} med={c['med']:.2f} div={c['div']:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
