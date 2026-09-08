"""MTQ-only (3+0) cells under the QP allocator at n=100 for both tasks, completing
the Table 3 row. Same construction as tune_planner.run_one_qp (FeedforwardQP MRO
mixin, campaign PD gains, settled bus, one orbit); the existing 30 boresight
trials are kept and seeds 30-99 added; full attitude runs seeds 0-99.
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import make_config, simulate, error_series, T_ORBIT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output_data", "wave")
PD_KP, PD_KD = 2.9e-4, 8.68e-3
CELLS = {"reduced": ("qp_0rw_reduced", "qp0rw_s{seed:04d}.pkl"),
         "full": ("qp_0rw_full", "qp0rw_full_s{seed:04d}.pkl")}


def _path(task, seed):
    d, f = CELLS[task]
    return os.path.join(OUT, d, f.format(seed=seed))


def _worker(job):
    task, seed = job
    if os.path.exists(_path(task, seed)):
        return job
    from ADCS.controller.mtq_w_rw_QP import MTQ_w_RW_QP
    from papers.IAC_1RW._feedforward import FeedforwardLP

    class FeedforwardQP(FeedforwardLP, MTQ_w_RW_QP):
        pass

    def maker(sat, config):
        return FeedforwardQP(est_sat=sat, p_gain=PD_KP, d_gain=PD_KD, c_gain=1e-3,
                             h_target=np.zeros(3), mode="dipole")
    config = dict(make_config(seed, n_rw=0, task=task, tf=T_ORBIT, dt=1.0, seed=seed),
                  controller="pd")
    r = simulate(config, maker, disturbances=("gg", "drag", "srp", "dipole", "general"),
                 bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
    os.makedirs(os.path.dirname(_path(task, seed)), exist_ok=True)
    tmp = f"{_path(task, seed)}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(r, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _path(task, seed))
    return job


def report():
    for task, (d, _) in CELLS.items():
        fin = np.array([float(error_series(pickle.load(open(p, "rb")))[-1])
                        for p in sorted(glob.glob(os.path.join(OUT, d, "*.pkl")))])
        if not len(fin):
            continue
        print(f"[qp 3+0 {task}] n={len(fin)} conv5 {100*np.mean(fin<=5):.0f}% conv1 {100*np.mean(fin<=1):.0f}% "
              f"median {np.median(fin):.1f} >30: {100*np.mean(fin>30):.0f}%")


def main():
    import multiprocessing as mp
    if "--report" not in sys.argv:
        jobs = [("reduced", s) for s in range(100)] + [("full", s) for s in range(100)]
        jobs = [j for j in jobs if not os.path.exists(_path(*j))]
        print(f"{len(jobs)} trials to run", flush=True)
        with mp.get_context("fork").Pool(processes=max(1, os.cpu_count() - 2), maxtasksperchild=1) as pool:
            for j in pool.imap_unordered(_worker, jobs):
                print("done", j, flush=True)
    report()
    print("qp cells finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
