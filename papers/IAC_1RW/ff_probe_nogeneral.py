"""Control arm for the dipole-cancellation probe: same seeds, estimator WITHOUT the lumped
general-torque state (the plant's general disturbance applies zero torque, so removing it
from the disturbance tuple changes only the filter). Compare against ff_probe20.py."""
import glob, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from papers.IAC_1RW._iac_sim import make_config, simulate, T_ORBIT, IAC_6U  # noqa: E402
from papers.IAC_1RW.generate_A_baseline import make_pd  # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "output_data", "ff_probe")
SEEDS = list(range(12))
def _path(s): return os.path.join(OUT, f"ffnogen_s{s:04d}.pkl")
def _worker(seed):
    if os.path.exists(_path(seed)): return seed
    cfg = dict(make_config(seed, n_rw=1, task="reduced", tf=T_ORBIT, dt=1.0, seed=seed), controller="pd")
    r = simulate(cfg, make_pd, use_estimator=True, disturbances=("gg", "drag", "srp", "dipole"),
                 bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
    keep = {"seed": seed, "dipole_est": np.asarray(r["dipole_est"], float), "h_wheel": np.asarray(r["state"], float)[:, 7],
            "time": np.asarray(r["time"], float), "final_deg": None}
    tmp = f"{_path(seed)}.tmp.{os.getpid()}"
    with open(tmp, "wb") as f: pickle.dump(keep, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, _path(seed)); return seed
def report():
    m = IAC_6U.m_res * np.asarray(IAC_6U.m_res_dir, float) / np.linalg.norm(IAC_6U.m_res_dir)
    print(f"{'seed':>4} {'with general: 25/50/75/final':>34}   {'dipole only: 25/50/75/final':>32}")
    for s in SEEDS:
        row = []
        for tag in ("ffprobe", "ffnogen"):
            p = os.path.join(OUT, f"{tag}_s{s:04d}.pkl")
            if not os.path.exists(p): row.append("   missing"); continue
            de = pickle.load(open(p, "rb"))["dipole_est"]; n = len(de); res = np.linalg.norm(de - m, axis=1) / np.linalg.norm(m)
            row.append(f"{res[int(.25*n)]:5.2f}/{res[int(.5*n)]:5.2f}/{res[int(.75*n)]:5.2f}/{res[int(.9*n):].mean():5.2f}")
        print(f"{s:4d} {row[0]:>34}   {row[1]:>32}")
    print("nogeneral probe finished", flush=True)
def main():
    import multiprocessing as mp
    if "--report" not in sys.argv:
        with mp.get_context("fork").Pool(processes=min(12, os.cpu_count() - 3), maxtasksperchild=1) as pool:
            for s in pool.imap_unordered(_worker, SEEDS): print("done", s, flush=True)
    report(); return 0
if __name__ == "__main__": sys.exit(main())
