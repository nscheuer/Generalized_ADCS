"""Figure 3 (headline): the one-wheel envelope. Built per the plotting spec +
FIG3_DATA_MAP amendments. Okabe-Ito palette, line-style redundancy, two panels.
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
OUT = os.path.join(HERE, "output_data")

T_ORBIT = 5553.6
H_MAX = 15e-3
TAU_PERP = 3.4e-6            # transverse disturbance budget, section IV-A
CEIL = 0.42                  # quadrature ceiling, measured (Campaign C)
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "grey": "#666666"}


def cell_stats(pkls):
    fin, dend = [], []
    for p in pkls:
        with open(p, "rb") as f:
            r = pickle.load(f)
        fin.append(float(error_series(r)[-1]))
        hf = np.asarray(r["h_frac"], float)
        dend.append(float(hf[-1]))
    fin = np.asarray(fin)
    conv = fin <= 30.0
    return (float(np.median(np.asarray(dend))), float(np.median(fin[conv])),
            float(100 * np.mean(~conv)))


def main():
    # --- markers from final data (converged-only medians; divergence % annotated) ---
    red_pd = cell_stats(glob.glob(os.path.join(OUT, "wave/pd_reduced_kp1/*.pkl")))
    red_pl = cell_stats(glob.glob(os.path.join(OUT, "A_trials/1rw_reduced_planner_seed*.pkl")))
    ful_pd = cell_stats(glob.glob(os.path.join(OUT, "wave/pd_full_kp1/*.pkl")))
    ful_pl = cell_stats(glob.glob(os.path.join(OUT, "tune_seed*_wave_planner_full.pkl")))
    d18 = json.load(open(os.path.join(OUT, "A_baseline_20260818_202627.json")))

    def jcell(k):
        h = d18["cells"][k]["horizons"]["5554"]
        dv = 100.0 - h["conv_pct_5deg"] if False else 100.0 - h["conv_pct_5deg"]
        dend = float(np.median(np.asarray(h["per_trial_h_frac_end"], float)))
        return (dend, float(h["median_final_deg"]), max(0.0, 100.0 - h["conv_pct_5deg"]))
    red_33 = jcell("3rw_reduced_pd")
    ful_33 = jcell("3rw_full_pd")

    # --- 6U at 400/600 from F; class bars from CDS + dipole range ---
    fj = json.load(open(os.path.join(OUT, "F_altitude_20260818_174558.json")))
    rows = {r["alt_km"]: r for r in fj["rows_by_case"]["m_res=0.05"]}
    D6_400 = rows[400.0]["accum_along_wheel_Nms"] / H_MAX
    D6_600 = rows[600.0]["accum_along_wheel_Nms"] / H_MAX
    src = rows[400.0]["accum_by_source_Nms"]
    dip6 = float(src.get("dipole", 1.29e-3))
    drag6 = float(src.get("drag", 1.42e-3))
    bars = {}
    for name, area_ratio, hmax_c in (("1U", 0.015 / 0.057, 3e-3),
                                     ("3U", 0.040 / 0.057, 15e-3)):
        lo = (drag6 * area_ratio + dip6 * (0.02 / 0.05)) / hmax_c
        hi = (drag6 * area_ratio + dip6 * (0.10 / 0.05)) / hmax_c
        bars[name] = (lo, hi)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, sharex=True)
    D = np.logspace(-2, 1, 200)
    tau_aT = 0.07 * H_MAX                       # reference along-wheel accum per orbit

    for ax, task, marks in (
            (axes[0], "Boresight-only (reduced attitude)",
             [("PD 3+1", red_pd, "o", OI["blue"]),
              ("planner 3+1", red_pl, "^", OI["orange"]),
              ("PD 3+3", red_33, "s", OI["green"])]),
            (axes[1], "Full three-axis attitude",
             [("PD 3+1", ful_pd, "o", OI["blue"]),
              ("planner 3+1 (tuned)", ful_pl, "^", OI["orange"]),
              ("PD 3+3", ful_33, "s", OI["green"])])):
        # B1 saturation
        ax.axvline(1.0, color=OI["verm"], lw=1.6, ls="-")
        ax.text(1.08, 1.4e-2, "wheel saturation  D = 1", rotation=90,
                fontsize=7.5, color=OI["verm"], va="bottom")
        # B2 pointing/drift (T_corr = 1 orbit) + fainter family (T/4); B3 dashed cut
        th_b2 = np.degrees(TAU_PERP * T_ORBIT * D / tau_aT)
        cut = np.degrees(TAU_PERP * T_ORBIT / (CEIL * H_MAX))
        ok = th_b2 >= cut
        ax.plot(D[ok], th_b2[ok], color=OI["grey"], lw=1.4,
                label="B2: bias-only drift, $T_{corr}$=1 orbit" if task.startswith("Bores") else None)
        ax.plot(D[~ok], th_b2[~ok], color=OI["grey"], lw=1.2, ls=":")
        ax.plot(D, th_b2 / 4.0, color=OI["grey"], lw=0.8, alpha=0.45)
        # feasible region: left of D=1, above B2
        ax.fill_between(D[D <= 1], np.maximum(th_b2[D <= 1], 1e-2), 1e2,
                        color=OI["blue"], alpha=0.07, lw=0)
        ax.text(1.3e-2, 6e1, "feasible\n(one wheel suffices)", fontsize=7.5,
                color=OI["blue"], alpha=0.9)
        # markers
        for lab, (d, th, dv), mk, col in marks:
            filled = dv < 0.5
            ax.plot(d, th, mk, ms=8, mfc=col if filled else "none", mec=col,
                    mew=1.6, label=lab)
            if dv >= 0.5:
                ax.annotate(f"{dv:.0f}% div.", (d, th), textcoords="offset points",
                            xytext=(7, 5), fontsize=7, color=col)
        ax.set_title(task, fontsize=10, pad=26)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1e-2, 1e1); ax.set_ylim(1e-2, 1e2)   # tighter DOWN (small theta at bottom)
        ax.grid(alpha=0.2, which="both", lw=0.4)
        ax.set_xlabel(r"$D = |\hat a\cdot\tau_{sec}|\,T_{orb}/h_{max}$")

    axes[0].set_ylabel(r"required pointing $\theta_{max}$ [deg]  (tighter $\downarrow$)")
    # wheel-class ticks (top axis, panel a): D scales as 1/h_max
    top = axes[0].secondary_xaxis("top")
    ticks = [(3e-3, "3"), (15e-3, "15"), (50e-3, "50")]
    top.set_xticks([0.07 * H_MAX / h for h, _ in ticks])
    top.set_xticklabels([f"{lab}" for _, lab in ticks], fontsize=7.5)
    top.set_xlabel(r"wheel capacity [mN·m·s] at reference disturbances",
                   fontsize=7, labelpad=2)
    # dimensional overlays (panel a)
    a = axes[0]
    a.plot(D6_400, 3.2e1, "D", ms=6, color="k", mfc="k")
    a.annotate("6U @400", (D6_400, 3.2e1), textcoords="offset points",
               xytext=(5, -11), fontsize=7)
    a.plot(D6_600, 6.2e1, "D", ms=6, color="k", mfc="w")
    a.annotate("6U @600", (D6_600, 6.2e1), textcoords="offset points",
               xytext=(-38, -3), fontsize=7)
    for i, (name, (lo, hi)) in enumerate(bars.items()):
        y = 1.35e1 * (1.7 ** i)
        a.plot([lo, hi], [y, y], "-", color="k", lw=2.2, solid_capstyle="butt")
        a.annotate(name, (hi, y), textcoords="offset points", xytext=(4, -2),
                   fontsize=7)
    # annotations that are not curves
    a.annotate("below ~262 km:\nD > 1 for the\nreference bus",
               xy=(3.1, 2.6e-2), fontsize=7, ha="center", color=OI["verm"])
    axes[1].annotate(
        "slew feasibility: $A=T_{slew}\\sqrt{\\tau_w\\bar\\sigma/(J\\Theta)}\\geq 2$"
        " (§IV-B)\nconstrains the slew spec, not this plane",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
        bbox=dict(fc="white", ec="0.7", lw=0.6))
    axes[0].annotate(
        "divergent minority: low-$\\sigma$-dwell draws,\nhigh inclination (§VI-E; screen scope)",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
        bbox=dict(fc="white", ec="0.7", lw=0.6))
    axes[0].legend(loc="center right", fontsize=7.5, framealpha=0.95,
                   bbox_to_anchor=(0.99, 0.72))

    fig.suptitle("The one-wheel envelope: saturation (B1), drift/pointing (B2, "
                 r"$T_{corr}$ = 1 orbit), quadrature ceiling (B3, dotted = unreachable)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig3_envelope.{ext}"), dpi=220)
    print("markers (D, conv-median, div%):")
    for n, v in (("red PD", red_pd), ("red PL", red_pl), ("red 3+3", red_33),
                 ("ful PD", ful_pd), ("ful PL", ful_pl), ("ful 3+3", ful_33)):
        print(f"  {n}: D={v[0]:.3f} th={v[1]:.2f} div={v[2]:.0f}%")
    print(f"6U D400={D6_400:.3f} D600={D6_600:.3f}; bars={bars}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
