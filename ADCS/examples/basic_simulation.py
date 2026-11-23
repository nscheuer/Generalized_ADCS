"""Propagate a freely rotating spacecraft for 30 seconds."""

import numpy as np

import ADCS


def main() -> None:
    satellite = ADCS.Satellite(J_0=np.diag([2.0, 1.5, 1.0]))
    initial_state = ADCS.State(
        w=np.deg2rad([0.0, 0.0, 1.0]),
        q=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    orbit = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 7.5, 0.0]),
    )

    results = ADCS.simulate(
        x=initial_state,
        satellite=satellite,
        os0=orbit,
        dt=1.0,
        tf=30.0,
    )
    final_state = results.first().state_hist[-1]
    print("Final angular velocity [rad/s]:", final_state.w)
    print("Final attitude quaternion [scalar, x, y, z]:", final_state.q)


if __name__ == "__main__":
    main()
