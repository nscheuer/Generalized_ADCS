"""Quickstart example: three magnetorquers and one reaction wheel.

Run from the repository root with::

    python papers/Package/paper/section_III/III_quickstart.py
"""

import matplotlib.pyplot as plt
import numpy as np

import ADCS


def main() -> ADCS.SimulationResults:
    """Build the spacecraft, simulate it, and show the main results."""
    # Actuators: 3 magnetorquers + 1 reaction wheel.
    acts = [ADCS.MTQ(axis, max_torque=0.2) for axis in np.eye(3)]
    acts += [
        ADCS.RW(
            axis=np.array([0, 0, 1]),
            max_torque=0.23e-3,
            J=5.7e-6,
            h=0.0,
            h_max=3.5e-3,
        )
    ]

    # Sensors: 3-axis magnetometer.
    sens = [ADCS.MTM(axis) for axis in np.eye(3)]

    # Spacecraft.
    sat = ADCS.Satellite(
        mass=4,
        J_0=np.diag([0.03, 0.03, 0.01]),
        actuators=acts,
        sensors=sens,
        boresight=np.array([0, 0, 1]),
    )

    # Initial state: [angular velocity, quaternion, wheel momentum].
    x_0 = np.array([0.01, -0.02, 0.01, 1, 0, 0, 0, 0.0])
    x_0 = ADCS.State.from_array(x_0)

    # Controller, goal, and orbit.
    ctrl = ADCS.controller.MTQ_w_RW_LP(
        est_sat=sat,
        p_gain=5e-5,
        d_gain=2e-3,
        c_gain=1e-3,
    )
    goal = ADCS.goals.Coordinate_Goal(lat=42.36, lon=-71.06, alt=0.0)
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([5000, 0, 5000]),
        V=np.array([0, 7.5, 0]),
    )

    results = ADCS.simulate(
        x=x_0,
        satellite=sat,
        controller=ctrl,
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
        layout=(1, 3),
        title="Quickstart: MTQs and Reaction Wheel",
    )
    plt.show()
    return results


if __name__ == "__main__":
    simulation_results = main()
    print(f"Completed {len(simulation_results.first().time_s)} simulation steps.")
