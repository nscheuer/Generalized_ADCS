"""Residual-dipole cancellation in the closed loop, measured with the corrected logger:
20 seeds of the 3+1 boresight PD cell (same construction as generate_A_baseline), one
orbit each. Reports the estimate residual |m_est - m_true| / |m_true| over the orbit and
the wheel momentum's secular slope over the second half-orbit.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import make_config, simulate, T_ORBIT, IAC_6U  # noqa: E402
from papers.IAC_1RW.generate_A_baseline import make_pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data", "ff_probe")
SEEDS = list(range(20))


def _path(seed):
    return os.path.join(OUT, f"ffprobe_s{seed:04d}.pkl")


def _worker(seed):
    if os.path.exists(_path(seed)):
        return seed
    cfg = dict(make_config(seed, n_rw=1, task="reduced", tf=T_ORBIT, dt=1.0, seed=seed), controller="pd")
    r = simulate(cfg, make_pd, use_estimator=True,
                 disturbances=("gg", "drag", "srp", "dipole", "general"),
                 bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
    keep = {"seed": seed, "dipole_est": np.asarray(r["dipole_est"], float),
            "h_wheel": np.asarray(r["state"], float)[:, 7], "time": np.asarray(r["time"], float),
            "h_frac": np.asarray(r["h_frac"], float)}
    os.makedirs(OUT, exist_ok=True)
    tmp = f"{_path(seed)}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _path(seed))
    return seed


def report():
    m_true = IAC_6U.m_res * np.asarray(IAC_6U.m_res_dir, float) / np.linalg.norm(IAC_6U.m_res_dir)
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "ffprobe_s*.pkl"))):
        r = pickle.load(open(p, "rb")); de = r["dipole_est"]; n = len(de)
        res = np.linalg.norm(de - m_true, axis=1) / np.linalg.norm(m_true)
        h = r["h_wheel"] * 1e3; t = r["time"][:len(h)]; k = len(h) // 2
        slope = np.polyfit(t[k:], h[k:], 1)[0] * T_ORBIT
        rows.append((r["seed"], res[int(0.25 * n)], res[int(0.5 * n)], res[int(0.75 * n)],
                     res[int(0.9 * n):].mean(), h[-1] - h[0], slope))
    if not rows:
        print("no probes on disk"); return
    a = np.array(rows, float)
    print(f"n={len(a)} seeds")
    for j, name in ((1, "residual @25%"), (2, "residual @50%"), (3, "residual @75%"), (4, "residual, final 10% mean")):
        print(f"  {name:26s} median {np.median(a[:, j]):.3f}  IQR {np.percentile(a[:, j], 25):.3f}-{np.percentile(a[:, j], 75):.3f}")
    print(f"  wheel dh over orbit [mN m s]  median {np.median(a[:, 5]):+.2f}  median|.| {np.median(np.abs(a[:, 5])):.2f}")
    print(f"  secular slope x T, 2nd half   median {np.median(a[:, 6]):+.2f}  median|.| {np.median(np.abs(a[:, 6])):.2f}  (uncancelled along-wheel projection 1.09)")
    print("ff probe finished", flush=True)


def main():
    import multiprocessing as mp
    if "--report" not in sys.argv:
        with mp.get_context("fork").Pool(processes=min(12, os.cpu_count() - 3), maxtasksperchild=1) as pool:
            for s in pool.imap_unordered(_worker, SEEDS):
                print("done", s, flush=True)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
