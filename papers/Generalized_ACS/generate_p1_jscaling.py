"""TASK 2 -- J-scaling validation (Paper 1 headline claim).

Claim (sec:II-D / V): the framework scales gains with inertia (k_p, k_d ~ ||J||) so
a tuned law ports across differently-sized spacecraft, performance ~invariant to J.
This isolates and tests it.

Method: hold the law (framework LP-PD) + scenario set fixed. Three inertias:
  J1 = diag(0.022,0.022,0.004)  (reference; canonical gains tuned here)
  J2 = diag(0.0314,0.0341,0.0100)  (BeaverCube)
  J3 = 4 x J1                       (clearly larger)
For each, run the MC twice: SCALED gains (k *= trace(J)/trace(J1)) vs UNSCALED
(canonical, the same for all J). Scaled should be ~invariant; unscaled should
degrade as J grows (gains too small -> sluggish -> misses the 1000 s horizon).

Config 3MTQ+3RW (fully actuated, so controllability is not the limiter and the
gain-scaling effect is clean), full-attitude task. 100 paired trials.

PAPER1_SCALE (fast=smoke, paper=100). Emits:
  output_data/tab_jscaling.{tex,csv}, fig_jscaling.{png,pdf}, JSCALING_RESULTS.md
"""
import os, sys, json
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ADCS.controller import MTQ_w_RW_LP
from ADCS.helpers import metrics as M
from ADCS.helpers.math_helpers import normalize
from ADCS.mc.monte_carlo_runner import MonteCarloRunner
from papers.Generalized_ACS._paper1_sim import scale, simulate, make_config

OUT = "papers/Generalized_ACS/output_data"
CONV = 5.0
KP, KD, KC = 5e-5, 1e-3, 1e-3                 # canonical gains, tuned for J1
CONFIG, TASK = "3MTQ+3RW", "full"
J1 = np.diagflat([0.022, 0.022, 0.004])
INERTIAS = {"J1 (ref)": J1,
            "J2 (BeaverCube)": np.diagflat([0.0314, 0.0341, 0.0100]),
            "J3 (20x J1)": 20.0 * J1}    # clearly-larger bus; unscaled gains 20x too small
TRACE1 = float(np.trace(J1))


def goal_quat_for(rid):
    return normalize(np.random.default_rng(20_000 + rid).standard_normal(4))


def _make_controller(sat, config):
    kp, kd, kc = config["_gains"]
    return MTQ_w_RW_LP(est_sat=sat, p_gain=kp, d_gain=kd, c_gain=kc,
                       h_target=np.zeros(3))


def _worker(config):
    return simulate(config, _make_controller)   # module-level (picklable for MC pool)


def err_series(run):
    qr = np.asarray(run["config"]["goal_quat"], float)
    q = np.asarray(run["state"], float)[:, 3:7]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    qr = qr / np.linalg.norm(qr)
    return np.asarray(run["time"], float), np.rad2deg(2*np.arccos(np.clip(np.abs(q @ qr), 0, 1)))


def saturation(run):
    a = run.get("alpha")
    if a is None:
        return float("nan")
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return float(1.0 - np.mean(a)) if a.size else float("nan")


def run_cell(J, kp, kd, kc, n, tf, dt):
    Jl = J.tolist(); gains = (kp, kd, kc)
    def gen(rid, _Jl=Jl, _g=gains):
        c = make_config(rid, CONFIG, tf, dt, seed=rid)
        c["J_0"] = _Jl
        c["_gains"] = _g
        c["goal_quat"] = goal_quat_for(rid)
        return c
    runner = MonteCarloRunner(sim_func=_worker, config_generator=gen, num_runs=n)
    res = [r for r in runner.run() if r is not None]
    if not res:
        return dict(n=0, conv=float("nan"), mean=float("nan"), p95=float("nan"), sat=float("nan"))
    finals, sats = [], []
    for r in res:
        _, e = err_series(r); finals.append(float(e[-1]))
        sv = saturation(r)
        if np.isfinite(sv): sats.append(sv)
    finals = np.asarray(finals)
    return dict(n=int(finals.size), conv=float(100*np.mean(finals < CONV)),
                mean=float(np.mean(finals)), p95=float(np.percentile(finals, 95)),
                sat=float(np.mean(sats)) if sats else float("nan"))


def main():
    s = scale(); tf, dt, n = s["tf"], s["dt"], s["num_runs"]
    print(f"[T2 jscaling] {CONFIG} {TASK} n={n} tf={tf} dt={dt}")
    rows = []
    for name, J in INERTIAS.items():
        f = float(np.trace(J) / TRACE1)               # gain scale factor
        kp_s, kd_s, kc_s = KP*f, KD*f, KC*f
        scaled = run_cell(J, kp_s, kd_s, kc_s, n, tf, dt)
        unscaled = run_cell(J, KP, KD, KC, n, tf, dt)
        rows.append(dict(name=name, trace=float(np.trace(J)), scale=f,
                         kp_scaled=kp_s, scaled=scaled, unscaled=unscaled))
        print(f"  {name:16s} (trace {np.trace(J):.4f}, x{f:.2f}):  "
              f"SCALED conv {scaled['conv']:.0f}% mean {scaled['mean']:.2f}  1-a {scaled['sat']:.3f}  |  "
              f"UNSCALED conv {unscaled['conv']:.0f}% mean {unscaled['mean']:.2f}")

    # ---- figure: conv% and mean vs inertia, scaled vs unscaled ----
    names = [r["name"] for r in rows]; x = np.arange(len(rows)); w = 0.38
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 4))
    a0.bar(x-w/2, [r["scaled"]["conv"] for r in rows], w, color="#2c7fb8", label="scaled gains")
    a0.bar(x+w/2, [r["unscaled"]["conv"] for r in rows], w, color="#de2d26", label="unscaled (canonical)")
    a0.set_ylabel("converged [%  <5°]"); a0.set_ylim(0, 108); a0.set_title("Convergence vs inertia")
    a0.legend(fontsize=8)
    a1.bar(x-w/2, [r["scaled"]["mean"] for r in rows], w, color="#2c7fb8", label="scaled")
    a1.bar(x+w/2, [r["unscaled"]["mean"] for r in rows], w, color="#de2d26", label="unscaled")
    a1.axhline(CONV, ls="--", c="k", lw=0.8); a1.set_ylabel("mean final error [deg]")
    a1.set_title("Mean final error vs inertia")
    for a in (a0, a1):
        a.set_xticks(x); a.set_xticklabels(names, fontsize=8, rotation=12); a.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_jscaling.png", dpi=150); fig.savefig(f"{OUT}/fig_jscaling.pdf")
    plt.close(fig)

    with open(f"{OUT}/tab_jscaling.tex", "w") as f:
        f.write("\\begin{tabular}{l r r r r r}\n\\toprule\n")
        f.write("Inertia & $\\mathrm{tr}\\,J$ & gain$\\times$ & Conv\\% (scaled) & Mean$^\\circ$ (scaled) & Conv\\% (unscaled) \\\\\n\\midrule\n")
        for r in rows:
            f.write(f"{r['name']} & {r['trace']:.4f} & {r['scale']:.2f} & {r['scaled']['conv']:.0f} & "
                    f"{r['scaled']['mean']:.2f} & {r['unscaled']['conv']:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(f"{OUT}/tab_jscaling.csv", "w") as f:
        f.write("inertia,trace,gain_scale,conv_scaled,mean_scaled,1ma_scaled,conv_unscaled,mean_unscaled\n")
        for r in rows:
            f.write(f"{r['name']},{r['trace']:.5f},{r['scale']:.3f},{r['scaled']['conv']:.1f},"
                    f"{r['scaled']['mean']:.3f},{r['scaled']['sat']:.3f},{r['unscaled']['conv']:.1f},{r['unscaled']['mean']:.3f}\n")
    json.dump(rows, open(f"{OUT}/P1_jscaling.json", "w"), indent=2)

    inv = max(r["scaled"]["conv"] for r in rows) - min(r["scaled"]["conv"] for r in rows)
    degrades = rows[-1]["unscaled"]["conv"] < rows[0]["unscaled"]["conv"] - 10
    with open(f"{OUT}/JSCALING_RESULTS.md", "w") as f:
        f.write("# Task 2 -- J-scaling validation (3MTQ+3RW, full attitude)\n\n")
        f.write("Same law (framework LP-PD), same scenarios; gains scaled by trace(J)/trace(J1).\n\n")
        f.write("| Inertia | tr J | gain x | conv% scaled | mean° scaled | conv% UNSCALED | mean° UNSCALED |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['name']} | {r['trace']:.4f} | {r['scale']:.2f} | {r['scaled']['conv']:.0f} | "
                    f"{r['scaled']['mean']:.2f} | {r['unscaled']['conv']:.0f} | {r['unscaled']['mean']:.2f} |\n")
        f.write(f"\nScaled-gain convergence spread across inertias: {inv:.0f} points "
                f"({'INVARIANT -> validates the claim' if inv <= 10 else 'NOT invariant'}). "
                f"Unscaled gains {'DEGRADE on the larger inertia (canonical gains too small) -> the scaling is load-bearing' if degrades else 'do not clearly degrade -> contrast weak'}.\n")
    print(f"[T2] scaled-conv spread {inv:.0f}pts | unscaled degrades={degrades} | wrote tab/fig/JSCALING_RESULTS.md")


if __name__ == "__main__":
    main()
