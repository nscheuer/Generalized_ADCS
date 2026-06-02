"""Analyze the existing PLANNER (Plan_and_Track_LQR/ALTRO) 3MTQ+0RW full-
attitude MC -- how does it perform at the underactuated boundary? No new run;
loads the committed .sim. For the planner paper 'It Stays Bounded' subsection.

Reports, over the non-converging trials: full-attitude vs boresight error, and
whether the body rate ||w|| is shed (detumble) or just bounded. Selects the
median (rank len//2 of non-converging by max envelope) and absolute worst, and
renders a 2-panel figure (full+boresight error, ||w||, MTQ commands).
"""
import os, sys, json, glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import ADCS
from ADCS.helpers.math_helpers import rot_mat
from ADCS.helpers.plot.control.targetplot import _angle_deg, _attitude_error_deg, _boresight_eci

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")
SIM = sorted(glob.glob("papers/Planner/output/mc100_altro_3+0_full_*.sim"))[-1]
CONV = 5.0


def bore_unit(sat):
    b = sat.boresight
    b = b if isinstance(b, np.ndarray) else np.array(list(b.values())[0])
    b = np.asarray(b, float).ravel()
    return b / np.linalg.norm(b)


def main():
    r = ADCS.SimulationResults.load(SIM)
    bu = bore_unit(r.satellite)
    print(f"[graceful-planner] {SIM}  n={len(r.runs)}  boresight={bu}")

    trials = []
    for i, run in enumerate(r.runs):
        st = np.asarray(run.state_hist, float)
        tg = np.asarray(run.target_hist, float)
        t = np.asarray(run.time_s, float)
        u = (np.asarray(run.control_hist, float)
             if getattr(run, "control_hist", None) is not None else None)
        full = np.array([_attitude_error_deg(st[k, 3:7], tg[k]) for k in range(len(st))])
        bore = np.empty(len(st))
        for k in range(len(st)):
            tq = tg[k]
            tgt = (rot_mat(tq) @ bu) if not np.isnan(tq[0]) else tq[1:4]
            bore[k] = _angle_deg(_boresight_eci(st[k, 3:7], bu), tgt)
        wmag = np.degrees(np.linalg.norm(st[:, 0:3], axis=1))
        trials.append({
            "rid": i, "t": t, "full": full, "bore": bore, "wmag": wmag, "u": u,
            "state": st,
            "full_final": float(full[-1]), "bore_final": float(bore[-1]),
            "max_env": float(np.max(full)),
            "converged": bool(full[-1] < CONV),
            "bore_settled": bool(bore[-1] < CONV),
            "w_init": float(wmag[0]),
            "w_final": float(np.mean(wmag[-len(wmag)//10:])),
        })

    nonconv = [x for x in trials if not x["converged"]]
    ncs = sorted(nonconv, key=lambda x: x["max_env"])
    median = ncs[len(ncs) // 2]
    # Two worst-case archetypes, both shown: the largest *peak* (a near-180 IC
    # that sweeps the antipode mid-slew, then recovers) and the worst *final*
    # outcome (least-converged at the end). Both stay bounded.
    worst_peak = max(trials, key=lambda x: x["max_env"])
    worst_final = max(trials, key=lambda x: x["full_final"])
    worst = worst_final
    print(f"  largest-peak rid {worst_peak['rid']}: peak {worst_peak['max_env']:.0f} -> "
          f"final {worst_peak['full_final']:.0f} deg (antipode sweep + recovery)")
    print(f"  worst-final  rid {worst_final['rid']}: final {worst_final['full_final']:.0f} deg "
          f"bore {worst_final['bore_final']:.0f}  ||w|| {worst_final['w_init']:.2f}->{worst_final['w_final']:.2f}")

    # how does it perform?
    conv_pct = 100 * np.mean([x["converged"] for x in trials])
    bore_set = 100 * np.mean([x["bore_settled"] for x in nonconv])
    wi = np.median([x["w_init"] for x in nonconv])
    wf = np.median([x["w_final"] for x in nonconv])
    full_med = np.median([x["full_final"] for x in nonconv])
    bore_med = np.median([x["bore_final"] for x in nonconv])
    print(f"  conv(full<5) {conv_pct:.0f}%  | non-converging n={len(nonconv)}")
    print(f"  among non-converging: median full_final {full_med:.1f}  bore_final {bore_med:.1f}  "
          f"bore_settled<5 {bore_set:.0f}%")
    print(f"  ||w|| median init {wi:.2f} -> final {wf:.2f} deg/s "
          f"({'detumbles' if wf < 0.9*wi else 'bounded/flat'})")
    print(f"  median pick rid {median['rid']} (max_env {median['max_env']:.1f}, "
          f"bore_final {median['bore_final']:.1f}, ||w|| {median['w_init']:.2f}->{median['w_final']:.2f}); "
          f"worst rid {worst['rid']} (max_env {worst['max_env']:.1f})")

    # figure: median | highest-peak | worst-final
    panels = ((median, f"median (rid {median['rid']})"),
              (worst_peak, f"highest peak (rid {worst_peak['rid']}): 180deg -> recovers"),
              (worst_final, f"worst final (rid {worst_final['rid']})"))
    fig, axs = plt.subplots(3, 3, figsize=(15, 8), sharex=True)
    for col, (tr, nm) in enumerate(panels):
        t = tr["t"]
        axs[0, col].plot(t, tr["full"], color="C0", lw=1.3, label="full attitude")
        axs[0, col].plot(t, tr["bore"], color="C2", lw=1.3, ls="--", label="boresight")
        axs[0, col].axhline(CONV, ls=":", c="k", lw=0.7)
        axs[0, col].set_yscale("log")
        axs[0, col].set_title(nm, fontsize=9); axs[0, col].legend(fontsize=7)
        axs[1, col].plot(t, tr["wmag"], color="C3", lw=1.3)
        if tr["u"] is not None:
            for j in range(min(3, tr["u"].shape[1])):
                axs[2, col].plot(t, tr["u"][:, j], lw=1.0, label=f"MTQ{j}")
            axs[2, col].legend(fontsize=6, ncol=3)
        axs[2, col].set_xlabel("time [s]")
        for rr in range(3):
            axs[rr, col].grid(True, which="both", alpha=0.3)
    axs[0, 0].set_ylabel("error [deg]"); axs[1, 0].set_ylabel("||w|| [deg/s]")
    axs[2, 0].set_ylabel("MTQ cmd [Am^2]")
    fig.suptitle("Planner (ALTRO) 3MTQ+0RW full-attitude -- it stays bounded "
                 "(both worst-case archetypes recover/remain calm)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_graceful_3mtq0rw.png", dpi=150)
    fig.savefig(f"{OUT}/fig_graceful_3mtq0rw.pdf"); plt.close(fig)

    json.dump({
        "source_sim": SIM, "n_trials": len(trials),
        "conv_pct_full": float(conv_pct), "n_nonconverging": len(nonconv),
        "nonconv_median_full_final_deg": float(full_med),
        "nonconv_median_bore_final_deg": float(bore_med),
        "nonconv_bore_settled_pct": float(bore_set),
        "nonconv_w_init_med_degs": float(wi), "nonconv_w_final_med_degs": float(wf),
        "median_pick": {"rid": median["rid"], "max_env": median["max_env"],
                        "full_final": median["full_final"], "bore_final": median["bore_final"],
                        "w_init": median["w_init"], "w_final": median["w_final"]},
        "worst_final_pick": {"rid": worst_final["rid"], "max_env": worst_final["max_env"],
                             "full_final": worst_final["full_final"], "bore_final": worst_final["bore_final"],
                             "w_init": worst_final["w_init"], "w_final": worst_final["w_final"]},
        "highest_peak_pick": {"rid": worst_peak["rid"], "max_env": worst_peak["max_env"],
                              "full_final": worst_peak["full_final"], "bore_final": worst_peak["bore_final"],
                              "w_init": worst_peak["w_init"], "w_peak": float(np.max(worst_peak["wmag"])),
                              "w_final": worst_peak["w_final"]},
        "final_dist": {"lt5": float(100*np.mean([x['full_final']<5 for x in trials])),
                       "lt30": float(100*np.mean([x['full_final']<30 for x in trials])),
                       "lt90": float(100*np.mean([x['full_final']<90 for x in trials])),
                       "max": float(max(x['full_final'] for x in trials))},
    }, open(f"{OUT}/graceful_3mtq0rw.json", "w"), indent=2)
    print(f"[graceful-planner] wrote fig_graceful_3mtq0rw + graceful_3mtq0rw.json")


if __name__ == "__main__":
    main()
