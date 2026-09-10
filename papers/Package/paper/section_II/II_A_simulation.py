"""Run the paper example with the PyPI Generalized_ADCS 0.1.8 package.

Install the pinned dependency first with::

    python -m pip install -r papers/Package/requirements.txt

Run this file from the repository root with::

    python papers/Package/paper/section_II/II_A_simulation.py
"""

import numpy as np
import matplotlib.pyplot as plt

import ADCS


EXPECTED_VERSION = "0.1.8"


def main() -> ADCS.SimulationResults:
    """Build the spacecraft scenario and run the simulation."""
    if ADCS.__version__ != EXPECTED_VERSION:
        raise RuntimeError(
            f"This paper script requires Generalized_ADCS=={EXPECTED_VERSION}; "
            f"found {ADCS.__version__!r}."
        )

    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    x_0 = ADCS.State.from_array(np.array([0, 0, 0, 1, 0, 0, 0, 0]))

    ctrl = ADCS.controller.MTQ_w_RW_LP(
        est_sat=real_sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.zeros(3),
    )
    est = None  # State estimation is optional for this example.
    goal = ADCS.goals.ECI_Goal(eci_vector=np.array([1, 0, 0]))
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([-7000, 0, 0]),
        V=np.array([0, 7.5, 0]),
    )

    results = ADCS.simulate(
        x=x_0,
        satellite=real_sat,
        controller=ctrl,
        estimator=est,
        goal=goal,
        os0=os0,
        dt=1.0,
        tf=1000.0,
    )

    ADCS.plot(
        results,
        ADCS.plots.TargetPlot(title="Pointing Error", modes=["real_target"]),
        ADCS.plots.QuaternionPlotCombined(
            title="Orientation (quaternion)",
            sources=["real"],
        ),
        ADCS.plots.ControlPlotCombined(
            title="Control Effort",
            units="actuator units",
        ),
        layout=(3, 1),
        figsize=(10, 10),
        title="Section II — Fixed ECI Goal",
    )
    plt.show()
    return results


if __name__ == "__main__":
    simulation_results = main()
    print(f"Completed {len(simulation_results.first().time_s)} simulation steps.")
