"""Generalized ADCS 2.x runner (run inside the 2.x virtual environment).

2.x propagates the orbit first with its fixed-step RK4 ``Orbit`` class (J2 via
``zonal_J=2``), then integrates attitude step by step with SciPy's adaptive
RK45 (rtol = atol = 1e-7). It has no fixed-step RK4 option for attitude, so its
attitude uses RK45. Earth constants are set on the initial ``Orbital_State``
(2.x works in km), so they match the shared scenario exactly.

* "coupled": ``ADCS.simulate`` (orbit + attitude, every step recorded).
* "orbit": the ``Orbit`` propagation alone (2.x also evaluates its environment
  quantities for every recorded state as part of building the orbit).
"""

from __future__ import annotations

import io
import contextlib
import warnings
from importlib.metadata import version

import numpy as np

from common import S, emit, parse_args, timed_runs

warnings.filterwarnings("ignore")

import ADCS  # noqa: E402
from ADCS.orbits.orbit import Orbit  # noqa: E402

SEC2CENT = 1.0 / (86400.0 * 36525.0)


def initial_orbit():
    os0 = ADCS.Orbital_State(ephem=ADCS.Ephemeris(), J2000=0.22,
                             R=np.array(S.R0) / 1000.0, V=np.array(S.V0) / 1000.0)
    os0.mu_e = S.MU / 1e9          # km^3/s^2
    os0.R_e = S.R_EARTH / 1000.0   # km
    os0.J2coeff = S.J2
    return os0


def dcm_from_adcs2_q(q):
    # 2.x stores a scalar-first quaternion whose Hamilton rotation matrix maps
    # body to inertial; its transpose is the inertial-to-body DCM. (Checked
    # against the high-precision reference in test runs.)
    m = S.quat_b_from_i_to_dcm(q)
    return [[m[k][i] for k in range(3)] for i in range(3)]


def make_run_factory(scenario: str):
    def make_run(steps: int):
        satellite = ADCS.Satellite(mass=S.MASS, J_0=np.diag(S.INERTIA_DIAG))
        x0 = ADCS.State(w=np.array(S.W0), q=np.array(S.Q0))
        os0 = initial_orbit()
        tf = steps * S.DT

        def run():
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                if scenario == "orbit":
                    orbit = Orbit(os0, end_time=os0.J2000 + tf * SEC2CENT, dt=S.DT,
                                  zonal_J=2, fast=False, verbose=False)
                    final = orbit.get_os(J2000=float(orbit.times[-1]))
                    return {"r": (np.asarray(final.R) * 1000.0).tolist(),
                            "v": (np.asarray(final.V) * 1000.0).tolist(),
                            "t": (float(orbit.times[-1]) - os0.J2000) / SEC2CENT,
                            "recorded": len(orbit.times)}
                # ADCS.simulate(tf) records int(tf/dt) states ending at tf - dt, i.e.
                # it integrates one step fewer; ask for one more step so it
                # integrates exactly ``steps`` steps like the other tools.
                results = ADCS.simulate(x=x0, satellite=satellite, os0=os0, dt=S.DT, tf=tf + S.DT)
            run_ = results.first()
            xf, osf = run_.state_hist[-1], run_.os_hist[-1]
            return {"r": (np.asarray(osf.R) * 1000.0).tolist(),
                    "v": (np.asarray(osf.V) * 1000.0).tolist(),
                    "dcm_bi": dcm_from_adcs2_q(np.asarray(xf.q, dtype=float).tolist()),
                    "w": np.asarray(xf.w, dtype=float).tolist(),
                    "t": float(np.asarray(run_.time_s)[-1]),
                    "recorded": len(run_.state_hist)}
        return run
    return make_run


def main():
    args = parse_args()
    samples, final = timed_runs(make_run_factory(args.scenario), args)
    emit(tool="Generalized ADCS 2.x", variant="Generalized ADCS 2.x", version=version("generalized-adcs"),
         scenario=args.scenario, steps=args.steps, samples_s=samples, final=final,
         notes=["Orbit: fixed-step RK4 with J2; attitude: SciPy RK45 (rtol = atol = 1e-7) within each step.",
                "Orbit and attitude are integrated separately (orbit first)."])


if __name__ == "__main__":
    main()
