"""Check of the Section 4.2 sentence: 'a separate h0 = 0 run with the estimator
removed reaches the same shortfall'. Reruns the 12 diverged boresight PD seeds
of the wave cell with zero initial wheel momentum, (a) on the truth state
(use_estimator=False, as Campaigns B/C/D run) and (b) with the estimator kept,
everything else identical to generate_A_baseline's PD cell. Reports per seed:
final error, peak |h|/h_max, saturation, and the quadrature demand ratio.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import make_config, simulate, error_series, cell_metrics, T_ORBIT  # noqa: E402
from papers.IAC_1RW.generate_A_baseline import make_pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data", "h0_rerun")
SEEDS = [8, 12, 15, 16, 23, 29, 49, 53, 55, 62, 78, 85]
ARMS = {"h0zero_truth": False, "h0zero_est": True}


def _path(arm, seed):
    return os.path.join(OUT, f"{arm}_s{seed:04d}.pkl")


def _worker(job):
    arm, seed = job
    if os.path.exists(_path(arm, seed)):
        return job
    c = make_config(seed, n_rw=1, task="reduced", tf=T_ORBIT, dt=1.0, seed=seed)
    c["h0"] = np.array([0.0])
    r = simulate(c, make_pd, use_estimator=ARMS[arm],
                 disturbances=("gg", "drag", "srp", "dipole", "general"),
                 bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{_path(arm, seed)}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _path(arm, seed))
    return job


def report():
    for arm in ARMS:
        runs = [pickle.load(open(_path(arm, s), "rb")) for s in SEEDS if os.path.exists(_path(arm, s))]
        if not runs:
            continue
        m = cell_metrics(runs, T_ORBIT)
        qr = m.get("per_trial_quadrature_ratio")
        print(f"\n[{arm}] n={len(runs)}  conv5 {m['conv_pct_5deg']:.0f}%  median {m['median_final_deg']:.2f} deg")
        print(f"{'seed':>5} {'final':>8} {'peak h':>8} {'sat':>4} {'quad':>7}")
        for i, (s, r) in enumerate(zip([s for s in SEEDS if os.path.exists(_path(arm, s))], runs)):
            hf = np.asarray(r["h_frac"], float)
            print(f"{s:5d} {float(error_series(r)[-1]):8.2f} {hf.max():8.3f} {'Y' if hf.max() >= 0.999 else 'n':>4} "
                  f"{(qr[i] if qr is not None else float('nan')):7.2f}")
        hf_all = [float(np.asarray(r['h_frac'], float).max()) for r in runs]
        fin = np.array([float(error_series(r)[-1]) for r in runs])
        print(f"  diverged (>30): {int(np.sum(fin > 30))}/{len(runs)}; saturated: {sum(h >= 0.999 for h in hf_all)}/{len(runs)}; "
              f"median peak h {np.median(hf_all):.3f}")


def main():
    import multiprocessing as mp
    if "--report" not in sys.argv:
        jobs = [(a, s) for a in ARMS for s in SEEDS]
        os.makedirs(OUT, exist_ok=True)
        with mp.get_context("fork").Pool(processes=min(12, os.cpu_count() - 3), maxtasksperchild=1) as pool:
            for j in pool.imap_unordered(_worker, jobs):
                print("done", j, flush=True)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
