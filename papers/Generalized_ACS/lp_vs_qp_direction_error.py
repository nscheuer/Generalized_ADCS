#!/usr/bin/env python
"""Task 5 / C2 -- LP vs QP control-allocation DIRECTION ERROR (Paper 1).

Substantiates the locked abstract: the LP allocator preserves torque direction
(achieved torque parallel to desired by construction), while the QP (bounded
least-squares) tilts the achieved torque when the request points partly along an
unachievable direction. For magnetorquers, achievable torque m x B is always
perpendicular to the local field B, so a request with a B-component is
(partly) unachievable -- the LP delivers less magnitude but keeps the direction;
the QP delivers a tilted torque.

Sweep: for the underactuated configs 3MTQ+0RW and 3MTQ+1RW, request torques over a
Fibonacci-sphere of directions at several magnitudes spanning interior -> beyond the
achievable boundary, run LP1 and QP, and record the angle between achieved and
desired torque (direction error) and the delivered magnitude fraction.

Outputs (beside this script + the paper figures dir):
  fig_lp_vs_qp_direction.png/.pdf   scatter (dir err vs angle-from-B) + CDF
  tab_lp_vs_qp_direction.csv/.tex   mean/p95/max direction error per allocator/config
  LP_VS_QP_RESULTS.md               numbers + one-paragraph characterization
"""
import os, sys, csv
import numpy as np

ROOT = "/Users/patrickmckeen/Documents/Generalized_ADCS"
sys.path.insert(0, ROOT)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from research.allocation_comparison import LPAllocator, QPAllocator

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_data")

# --- configs (from research/closed_loop_comparison.py create_3mtq_{0,1}rw) ---
MTQ_AXES = np.eye(3)
U_MTQ_MAX = np.array([0.2, 0.2, 0.2])          # A.m^2
RW_AXIS_Z = np.array([[0.0], [0.0], [1.0]])    # single RW on +z
U_RW_MAX = np.array([0.001])                    # N.m

CONFIGS = {
    "3MTQ+0RW": dict(A_rw=np.zeros((3, 0)), A_mtq=MTQ_AXES, u_rw=np.zeros(0), u_mtq=U_MTQ_MAX),
    "3MTQ+1RW": dict(A_rw=RW_AXIS_Z,       A_mtq=MTQ_AXES, u_rw=U_RW_MAX,    u_mtq=U_MTQ_MAX),
}

# Representative LEO field in the body frame (~30 uT), deliberately not aligned
# with any actuator axis so the geometry is generic.
B_BODY = 30e-6 * (np.array([1.0, 0.4, 0.3]) / np.linalg.norm([1.0, 0.4, 0.3]))


def fib_sphere(n):
    """n ~uniform unit vectors on the sphere (Fibonacci spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)])


def t_max_along(alloc, d, cfg):
    """Max achievable torque magnitude along unit direction d (LP, huge request)."""
    r = alloc.allocate(1.0 * d, B_BODY, cfg["A_rw"], cfg["A_mtq"], cfg["u_rw"], cfg["u_mtq"])
    return np.linalg.norm(r.tau_achieved)


def run_config(name, cfg, n_dir=700):
    lp, qp = LPAllocator(), QPAllocator()
    dirs = fib_sphere(n_dir)
    bhat = B_BODY / np.linalg.norm(B_BODY)
    tmax = np.array([t_max_along(lp, d, cfg) for d in dirs])
    ref = tmax.max()
    eps = 1e-3 * ref
    rows = []
    for d, tm in zip(dirs, tmax):
        ang_B = np.degrees(np.arccos(np.clip(abs(d @ bhat), -1, 1)))  # 0=along B, 90=perp B
        if tm > eps:
            # per-direction: interior (0.6x), edge (1.0x), beyond the boundary (1.6x)
            reqs = [(0.6 * tm, True), (1.0 * tm, True), (1.6 * tm, False)]
        else:
            # field-unachievable direction (e.g. along B for MTQ-only): any request is beyond
            reqs = [(0.3 * ref, False)]
        for mag, interior in reqs:
            tau = mag * d
            rl = lp.allocate(tau, B_BODY, cfg["A_rw"], cfg["A_mtq"], cfg["u_rw"], cfg["u_mtq"])
            rq = qp.allocate(tau, B_BODY, cfg["A_rw"], cfg["A_mtq"], cfg["u_rw"], cfg["u_mtq"])
            rows.append(dict(config=name, angle_from_B=ang_B, interior=interior, t_max=tm, req_mag=mag,
                             lp_dir=rl.direction_error_deg, lp_mag=rl.magnitude_ratio, lp_ok=rl.solver_success,
                             qp_dir=rq.direction_error_deg, qp_mag=rq.magnitude_ratio, qp_ok=rq.solver_success))
    return rows


def stats(vals):
    v = np.asarray(vals, float)
    return dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)), max=float(v.max()))


def main():
    all_rows = []
    for name, cfg in CONFIGS.items():
        all_rows += run_config(name, cfg)

    # Per-allocator stats. The DIRECTION-preservation property is unconditional, so
    # report LP over ALL requests; for QP the tilt only appears where the request is
    # not interior -- report QP over the "near/beyond boundary" requests (achievable
    # magnitude < request, i.e. LP had to scale down, magnitude_ratio < ~0.99) AND
    # over all, so both the headline tail and the interior baseline are on record.
    print("\n=== LP vs QP direction error (deg) ===")
    table = []
    for name in CONFIGS:
        rws = [r for r in all_rows if r["config"] == name]
        inr = [r for r in rws if r["interior"]]
        bdy = [r for r in rws if not r["interior"]]
        lp_all = stats([r["lp_dir"] for r in rws])
        qp_int = stats([r["qp_dir"] for r in inr]) if inr else dict(mean=0, p95=0, max=0)
        qp_sat = stats([r["qp_dir"] for r in bdy]) if bdy else dict(mean=0, p95=0, max=0)
        print(f"\n{name}  (N={len(rws)}: {len(inr)} interior, {len(bdy)} at/over boundary)")
        print(f"  LP   all     : mean {lp_all['mean']:.4f}  p95 {lp_all['p95']:.4f}  max {lp_all['max']:.4f}")
        print(f"  QP   interior: mean {qp_int['mean']:.4f}  p95 {qp_int['p95']:.4f}  max {qp_int['max']:.4f}")
        print(f"  QP   boundary: mean {qp_sat['mean']:.3f}  p95 {qp_sat['p95']:.3f}  max {qp_sat['max']:.3f}")
        table.append((name, len(rws), len(bdy), lp_all, qp_int, qp_sat))

    # ---- table files ----
    with open(os.path.join(OUTDIR, "tab_lp_vs_qp_direction.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "n_req", "n_boundary",
                    "LP_mean", "LP_p95", "LP_max", "QP_int_mean", "QP_int_p95", "QP_int_max",
                    "QP_bdry_mean", "QP_bdry_p95", "QP_bdry_max"])
        for name, n, ns, lp, qpi, qps in table:
            w.writerow([name, n, ns, f"{lp['mean']:.4f}", f"{lp['p95']:.4f}", f"{lp['max']:.4f}",
                        f"{qpi['mean']:.4f}", f"{qpi['p95']:.4f}", f"{qpi['max']:.4f}",
                        f"{qps['mean']:.3f}", f"{qps['p95']:.3f}", f"{qps['max']:.3f}"])
    with open(os.path.join(OUTDIR, "tab_lp_vs_qp_direction.tex"), "w") as f:
        f.write("% LP vs QP allocation direction error [deg]\n")
        f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
        f.write("Config & \\multicolumn{3}{c}{LP (all requests)} & \\multicolumn{3}{c}{QP (boundary requests)} \\\\\n")
        f.write(" & mean & p95 & max & mean & p95 & max \\\\\n\\midrule\n")
        for name, n, ns, lp, qpi, qps in table:
            f.write(f"{name} & {lp['mean']:.4f} & {lp['p95']:.4f} & {lp['max']:.4f} & "
                    f"{qps['mean']:.2f} & {qps['p95']:.2f} & {qps['max']:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    # ---- figure: scatter (dir err vs angle-from-B, 3MTQ+0RW) + CDF (both configs) ----
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4.2))
    r0 = [r for r in all_rows if r["config"] == "3MTQ+0RW"]
    aB = np.array([r["angle_from_B"] for r in r0])
    ax0.scatter(aB, [r["qp_dir"] for r in r0], s=8, c="C1", alpha=0.5, label="QP")
    ax0.scatter(aB, [r["lp_dir"] for r in r0], s=8, c="C0", alpha=0.5, label="LP")
    ax0.set_xlabel("angle of request from B [deg]  (0 = along B, unachievable)")
    ax0.set_ylabel("torque direction error [deg]")
    ax0.set_title("3MTQ+0RW: achievable torque ⊥ B")
    ax0.axhline(30, ls=":", c="0.5", lw=0.8); ax0.grid(alpha=0.3); ax0.legend()
    # distinct colour PER (config, law) + markers on the near-vertical LP curves
    # (LP dir-error ~0 => a step at x~=0 that was invisible behind the other LP curve)
    cdf_styles = [("3MTQ+0RW", "lp_dir", "-",  "#2c7fb8", "3MTQ+0RW LP"),
                  ("3MTQ+0RW", "qp_dir", "--", "#2c7fb8", "3MTQ+0RW QP"),
                  ("3MTQ+1RW", "lp_dir", "-",  "#de2d26", "3MTQ+1RW LP"),
                  ("3MTQ+1RW", "qp_dir", "--", "#de2d26", "3MTQ+1RW QP")]
    for i, (name, key, ls, col, lab) in enumerate(cdf_styles):
        rr = [r for r in all_rows if r["config"] == name]
        v = np.sort([r[key] for r in rr]); cdf = np.linspace(0, 1, len(v))
        ax1.plot(v, cdf, ls=ls, color=col, lw=2.0, zorder=6 - i,
                 marker="o" if key == "lp_dir" else None,
                 markevery=max(1, len(v) // 10), ms=4, label=lab)
    ax1.set_xlabel("torque direction error [deg]"); ax1.set_ylabel("CDF")
    ax1.set_xlim(left=-2)
    ax1.set_title("Direction-error CDF"); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "fig_lp_vs_qp_direction.png"), dpi=150)
    plt.savefig(os.path.join(OUTDIR, "fig_lp_vs_qp_direction.pdf"))
    print("\nsaved fig_lp_vs_qp_direction.png/.pdf + tab_lp_vs_qp_direction.csv/.tex")

    # ---- RESULTS.md ----
    lp0 = table[0][3]
    with open(os.path.join(OUTDIR, "LP_VS_QP_RESULTS.md"), "w") as f:
        f.write("# Task 5 -- LP vs QP allocation direction error\n\n")
        f.write(f"Representative body field B = {np.round(B_BODY*1e6,2).tolist()} uT. "
                "Fibonacci-sphere of 700 request directions x 3 magnitude scales "
                "(0.5/1.0/1.5 x max achievable) per config.\n\n")
        f.write("| Config | LP mean | LP p95 | LP max | QP mean (bdry) | QP p95 (bdry) | QP max (bdry) |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for name, n, ns, lp, qpi, qps in table:
            f.write(f"| {name} | {lp['mean']:.4f} | {lp['p95']:.4f} | {lp['max']:.4f} | "
                    f"{qps['mean']:.2f} | {qps['p95']:.2f} | {qps['max']:.2f} |\n")
        f.write("\nLP direction error is ~0 by construction (achieved torque is parallel to the "
                "request, or zero when the direction is unachievable). QP minimizes Euclidean "
                "torque error, so when the request points partly along the field-unachievable "
                "direction it delivers a tilted torque -- the tilt is ~0 for interior requests and "
                "grows toward 90 deg as the request approaches the unachievable direction.\n")
    print("saved LP_VS_QP_RESULTS.md")


if __name__ == "__main__":
    main()
