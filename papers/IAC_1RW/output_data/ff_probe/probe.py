"""Full-orbit probe of the estimated-dipole feedforward: logs the filter's dipole state and
the est_sat main_param the feedforward reads, every step, for one seed."""
import sys, pickle, numpy as np; sys.path.insert(0, "/Users/patrickmckeen/ADCS_wt/iac-1rw")
from papers.IAC_1RW._iac_sim import make_config, simulate, T_ORBIT
from papers.IAC_1RW.generate_A_baseline import make_pd
from ADCS.estimators.attitude_estimators.attitude_UAKF import UAKF
from ADCS.satellite_hardware.disturbances import Dipole_Disturbance
seed = int(sys.argv[1]); log = []
_orig = UAKF.update
def patched(self, *a, **k):
    out = _orig(self, *a, **k)
    mp = [np.ravel(d.main_param)[:3].copy() for d in self.est_sat.disturbances if isinstance(d, Dipole_Disturbance)]
    log.append(mp[0]); return out
UAKF.update = patched
cfg = dict(make_config(seed, n_rw=1, task="reduced", tf=T_ORBIT, dt=1.0, seed=seed), controller="pd")
r = simulate(cfg, make_pd, use_estimator=True, disturbances=("gg","drag","srp","dipole","general"), bus_kwargs={"tau_w": 2.0e-3, "h_max": 15.0e-3})
pickle.dump({"seed": seed, "dipole_est": np.array(log), "h_frac": np.asarray(r["h_frac"]), "state7": np.asarray(r["state"])[:, 7], "time": np.asarray(r["time"])},
            open(f"/Users/patrickmckeen/ADCS_wt/iac-1rw/papers/IAC_1RW/output_data/ff_probe/probe_s{seed:04d}.pkl", "wb"))
print("done", seed, flush=True)
