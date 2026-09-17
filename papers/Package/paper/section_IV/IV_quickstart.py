"""Section IV quickstart example based on Tutorial 01.

This script intentionally uses the public ``Generalized_ADCS==0.1.8`` API
used by the paper package environment.  The matplotlib figure is written to
``section_IV/outputs`` before it is displayed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ADCS


EXPECTED_VERSION = "0.1.8"
OUTPUT_DIR = Path(__file__).with_name("outputs")


def _save_current_figure(name: str) -> None:
    """Save the figure most recently created by ``ADCS.plot``."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    plt.gcf().savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")


def main() -> ADCS.SimulationResults:
    """Run the Tutorial 01 scenario and save the matplotlib plot."""
    if ADCS.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"This paper script requires Generalized_ADCS=={EXPECTED_VERSION}; "
            f"found {ADCS.__version__!r}. Install "
            "papers/Package/requirements-package.txt."
        )

    acts = [
        ADCS.RW(
            axis=np.array([1.0, 0.0, 0.0]),
            max_torque=0.0023,
            J=5.7e-6,
            h=0.0,
            h_max=0.0036,
        )
    ]
    acts += [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
    sensors = [ADCS.MTM(axis) for axis in np.eye(3)]

    satellite = ADCS.Satellite(
        mass=4.0,
        J_0=np.diag([0.03, 0.03, 0.01]),
        actuators=acts,
        sensors=sensors,
        boresight=np.array([0.0, 0.0, 1.0]),
    )
    x_0 = ADCS.State.from_array(
        np.array([0.01, -0.02, 0.01, 1.0, 0.0, 0.0, 0.0, 0.0])
    )
    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5e-5,
        d_gain=2e-3,
        c_gain=1e-3,
        h_target=np.zeros(3),
    )
    goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.06, alt=0.0)
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([5000.0, 0.0, 5000.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )
    results = ADCS.simulate(
        x=x_0,
        satellite=satellite,
        controller=controller,
        goal=goal,
        os0=os0,
        dt=1.0,
        tf=500.0,
    )

    ADCS.plot(
        results,
        ADCS.plots.TargetPlot(),
        ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
        ADCS.plots.ControlPlotCombined(),
        ADCS.plots.ControlPlotSingle(index=0, title="RW Control Torque"),
        layout=(2, 2),
        title="Section IV Quickstart: MTQ and RW Control",
    )
    _save_current_figure("iv_quickstart_results.png")
    plt.show()
    return results


if __name__ == "__main__":
    simulation_results = main()
    print(f"Completed {len(simulation_results.first().time_s)} simulation steps.")
