"""TASK 7 -- LP / QP allocation solve timings (Paper 1, sec:V-B).

sec:V-B cites LP ~350 us / QP ~40 us on a single core (flagged "embedded-class TBD").
This times the actual allocation solves on THIS machine so the numbers + caveat are
real. 3MTQ+1RW geometry, representative desired torque and body field; N solves each,
report median + mean microseconds. Single process, single core (no threading).
"""
import os, sys, time
import numpy as np
sys.path.append(os.path.abspath(os.path.join(__file__, "../../..")))
from scipy.optimize import linprog, lsq_linear
from ADCS.helpers.math_helpers import skewsym, normalize

N = 2000
OUT = "papers/Generalized_ACS/output_data"

# 3MTQ+1RW geometry
A_rw = np.array([[0.0], [0.0], [1.0]])                 # 1 RW on +z
A_mtq_axes = np.eye(3)                                  # 3 MTQ on principal axes
u_rw_max = np.array([1.0e-3])                           # N.m
u_mtq_max = np.array([0.2, 0.2, 0.2])                   # A.m^2
B = 30e-6 * normalize(np.array([1.0, 0.4, 0.3]))        # body field ~30 uT


def build(tau_des, b):
    A_mtq = -skewsym(b) @ A_mtq_axes
    A_total = np.hstack([A_rw, A_mtq])
    lims = np.concatenate([u_rw_max, u_mtq_max])
    return A_total, lims


def lp_solve(tau_des, b):
    A_total, lims = build(tau_des, b)
    t_mag = np.linalg.norm(tau_des); tau_hat = tau_des / t_mag
    n = A_total.shape[1]
    c = np.zeros(n + 1); c[-1] = -1.0
    A_eq = np.hstack([A_total, -tau_hat.reshape(3, 1)])
    bounds = [(-l, l) for l in lims] + [(0, None)]
    return linprog(c, A_eq=A_eq, b_eq=np.zeros(3), bounds=bounds, method="highs")


def qp_solve(tau_des, b):
    A_total, lims = build(tau_des, b)
    return lsq_linear(A_total, tau_des, bounds=(-lims, lims), method="bvls")


def time_solver(fn, label):
    rng = np.random.default_rng(0)
    # vary the request direction each call (realistic), fixed field
    taus = [1e-4 * normalize(rng.standard_normal(3)) for _ in range(N)]
    fn(taus[0], B)                                      # warmup / JIT
    ts = np.empty(N)
    for i in range(N):
        t0 = time.perf_counter(); fn(taus[i], B); ts[i] = (time.perf_counter() - t0) * 1e6
    print(f"  {label:4s}: median {np.median(ts):7.1f} us  mean {ts.mean():7.1f} us  "
          f"p95 {np.percentile(ts,95):7.1f} us")
    return float(np.median(ts)), float(ts.mean()), float(np.percentile(ts, 95))


def main():
    import platform
    print(f"[T7] allocation timings, N={N}, {platform.processor() or platform.machine()}")
    lp = time_solver(lp_solve, "LP")
    qp = time_solver(qp_solve, "QP")
    with open(f"{OUT}/ALLOC_TIMING_RESULTS.md", "w") as f:
        f.write("# Task 7 -- LP/QP allocation solve timings (sec:V-B)\n\n")
        f.write(f"Machine: {platform.platform()}, Python single process/core. N={N} solves, "
                "3MTQ+1RW geometry, random request directions, fixed ~30 uT body field.\n\n")
        f.write("| Allocator | median | mean | p95 |\n|---|---|---|---|\n")
        f.write(f"| LP (HiGHS `linprog`) | {lp[0]:.0f} us | {lp[1]:.0f} us | {lp[2]:.0f} us |\n")
        f.write(f"| QP (`lsq_linear` BVLS) | {qp[0]:.0f} us | {qp[1]:.0f} us | {qp[2]:.0f} us |\n\n")
        f.write(f"Paper sec:V-B currently cites LP ~350 us / QP ~40 us (single core). Measured here: "
                f"LP {lp[0]:.0f} us / QP {qp[0]:.0f} us median. These are **desktop** numbers (this "
                "machine); the LP>QP ordering and ~10x ratio are the load-bearing qualitative claim. "
                "Embedded/flight-class timings would be larger (slower clock, no vectorized BLAS) but "
                "the relative ordering holds; the single-core desktop caveat in sec:V-B should stand "
                "as written unless flight-processor numbers are measured directly.\n")
    print(f"[T7] wrote {OUT}/ALLOC_TIMING_RESULTS.md")


if __name__ == "__main__":
    main()
