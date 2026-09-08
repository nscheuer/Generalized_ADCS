"""Review item 5: one failed trial three ways (seed 8) + PD finals histogram.
Panels: attitude error (log) / |h|/h_max / sigma / LP alpha, shared time axis.
Traces: PD (diverges), desaturation-first (exchanges failure mode), planner
(rescues). Okabe-Ito, line style redundant.
"""
import os
import pickle
import sys
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import error_series  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fig_style
fig_style.apply(8.0)
OUT = os.path.join(HERE, "output_data")
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00"}


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def threeway(runs, with_alpha, base):
    """Error / |h|/h_max / sigma (+ LP alpha if with_alpha), shared time axis."""
    fig_style.apply(base)
    sm = base - 2.5          # small annotation text
    n = 4 if with_alpha else 3
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 6.2 if with_alpha else 5.0),
                             sharex=True, constrained_layout=True)
    for name, r, col, ls in runs:
        t = np.asarray(r["time"], float) / 3600.0
        e = np.asarray(error_series(r), float)
        hf = np.asarray(r["h_frac"], float)
        sg = np.asarray(r["sigma"], float)
        k = min(len(t), len(e))
        axes[0].plot(t[:k], np.maximum(e[:k], 1e-3), color=col, ls=ls, lw=1.3,
                     label=f"{name} (final {e[-1]:.1f}°)")
        axes[1].plot(t[:len(hf)], hf, color=col, ls=ls, lw=1.3)
        axes[2].plot(t[:len(sg)], sg, color=col, ls=ls, lw=1.1)
        if with_alpha:
            alc = np.clip(np.asarray(r["alpha"], float), 0, 1)
            axes[3].plot(t[:len(alc)], alc, color=col, ls="-", lw=0.5, alpha=0.18)
            k60 = 61
            roll = np.convolve(alc, np.ones(k60) / k60, mode="same")
            axes[3].plot(t[:len(roll)], roll, color=col, ls=ls, lw=1.4)

    axes[0].set_yscale("log"); axes[0].set_ylim(3e-2, 2e2)
    axes[0].set_ylabel("attitude error [deg]")
    axes[0].legend(fontsize=sm, loc="lower left", framealpha=0.95)
    axes[1].axhline(1.0, color="0.3", lw=0.9, ls=":")
    axes[1].text(0.995, 1.02, "$h_{max}$", fontsize=sm, ha="right",
                 transform=axes[1].get_yaxis_transform())
    axes[1].set_ylabel(r"$|h|/h_{max}$"); axes[1].set_ylim(0, 1.1)
    axes[2].axhline(0.2, color="0.3", lw=0.9, ls=":")
    axes[2].text(0.34, 0.24, "dump-favorable threshold: $\\sigma=0.2$", fontsize=sm, ha="center",
                 transform=axes[2].get_yaxis_transform())
    axes[2].set_ylabel(r"$\sigma$"); axes[2].set_ylim(0, 1)
    if with_alpha:
        axes[3].set_ylabel(r"LP scale $\alpha$" + "\n(60 s median; raw faint)")
        axes[3].set_ylim(-0.05, 1.05)
    axes[-1].set_xlabel("time [hr]  (one orbit)")
    for ax, lab in zip(axes, "abcd"):
        ax.text(0.01, 0.97, f"({lab})", transform=ax.transAxes, fontsize=base + 0.5,
                fontweight="bold", va="top")
    return fig


def main():
    runs = [
        ("PD", load(os.path.join(OUT, "wave/pd_reduced_kp1/pd_reduced_kp1_s0008.pkl")),
         OI["blue"], "-"),
        ("desaturation-first", load(os.path.join(OUT, "seed8_reserved.pkl")),
         OI["verm"], "--"),
        ("planner", load(os.path.join(OUT, "A_trials/1rw_reduced_planner_seed0008.pkl")),
         OI["orange"], "-."),
    ]
    # full-width placement: three panels at body-size type is the default; the
    # four-panel (LP alpha) version is kept alongside in case it is preferred
    for with_alpha, stem, base in ((False, "fig_seed8_threeway", 10.0),
                                   (True, "fig_seed8_threeway_4panel", 8.0)):
        fig = threeway(runs, with_alpha, base)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), dpi=220)
        plt.close(fig)
    fig_style.apply(8.0)

    # histogram of PD-reduced finals, log x
    fin = [float(error_series(load(p))[-1])
           for p in sorted(glob.glob(os.path.join(OUT, "wave/pd_reduced_kp1/*.pkl")))]
    fig2, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    bins = np.logspace(np.log10(0.05), np.log10(200), 28)
    ax.hist(fin, bins=bins, color=OI["blue"], alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(5, color="0.4", lw=0.9, ls=":")
    ax.axvline(30, color=OI["verm"], lw=0.9, ls=":")
    ax.text(5, ax.get_ylim()[1]*0.92, " 5°", fontsize=7, color="0.35")
    ax.text(30, ax.get_ylim()[1]*0.92, " 30°", fontsize=7, color=OI["verm"])
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    fin = np.asarray(fin)
    n_lo = int(np.sum(fin < 5)); n_hi = int(np.sum(fin > 30))
    assert n_lo + n_hi == len(fin), (n_lo, n_hi, len(fin))   # nothing between 5 and 30 deg
    ax.text(1.6, ax.get_ylim()[1]*0.75, f"{n_lo} below 5°", fontsize=7.5, ha="center", color="0.25")
    ax.text(75, ax.get_ylim()[1]*0.75, f"{n_hi} above 30°", fontsize=7.5, ha="center", color="0.25")
    ax.set_xlabel("final pointing error [deg]"); ax.set_ylabel("trials")
    for ext in ("pdf", "png"):
        fig2.savefig(os.path.join(OUT, f"fig_pd_hist.{ext}"), dpi=220)
    print("finals seed8:", {n: round(float(error_series(r)[-1]), 2) for n, r, _, _ in runs})
    return 0


if __name__ == "__main__":
    sys.exit(main())
