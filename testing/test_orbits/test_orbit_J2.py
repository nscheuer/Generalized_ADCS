import numpy as np
import matplotlib.pyplot as plt

from ADCS.orbits.universal_constants import EarthConstants

from testing.test_orbits._helpers import make_reference_orbital_state


def closest_approach(reference, trajectory, min_skip=1):
    d2 = np.sum((trajectory - reference) ** 2, axis=1)
    if min_skip > 0:
        d2[:min_skip] = np.inf
    i_star = np.argmin(d2)
    return i_star, np.sqrt(d2[i_star])


def run_orbit(method="rk4", use_j2=True, dt=60.0):
    orbit = make_reference_orbital_state()
    mu = EarthConstants.mu_e
    r_mag = np.linalg.norm(orbit.R)
    t_orbit = 2.0 * np.pi * np.sqrt(r_mag**3 / mu)
    steps = int(t_orbit / dt)
    dt = t_orbit / steps
    times = np.linspace(0.0, t_orbit, steps + 1)

    positions = np.zeros((steps + 1, 3))
    positions[0] = orbit.R

    for i in range(steps):
        if method == "rk4":
            orbit = orbit.propagate_orbit_rk4(dt, J2_perturbation_on=use_j2, fast=True)
        elif method == "euler":
            orbit = orbit.propagate_orbit(dt, J2_perturbation_on=use_j2, fast=True)
        else:
            raise ValueError(f"Unknown method: {method}")
        positions[i + 1] = orbit.R

    return times, positions


def test_j2_trajectory_remains_bounded_over_one_orbit():
    _, positions = run_orbit(method="rk4", use_j2=True, dt=30.0)
    radii = np.linalg.norm(positions, axis=1)

    assert np.all(np.isfinite(positions))
    assert np.min(radii) > EarthConstants.R_e
    assert np.max(radii) < 1.2 * np.linalg.norm(positions[0])


def test_j2_trajectory_differs_from_two_body_trajectory():
    _, no_j2 = run_orbit(method="rk4", use_j2=False, dt=60.0)
    _, with_j2 = run_orbit(method="rk4", use_j2=True, dt=60.0)

    assert np.linalg.norm(with_j2[-1] - no_j2[-1]) > 1e-3


def test_j2_closest_approach_is_finite_and_not_initial_point():
    _, positions = run_orbit(method="rk4", use_j2=True, dt=10.0)

    i_star, d_min = closest_approach(positions[0], positions, min_skip=5)

    assert 5 <= i_star < len(positions)
    assert np.isfinite(d_min)
    assert d_min > 0.0


def main():
    times, positions = run_orbit(method="rk4", use_j2=True, dt=10.0)
    i_star, d_min = closest_approach(positions[0], positions, min_skip=5)

    print(f"Closest approach at index {i_star} / {len(positions) - 1}")
    print(f"Closest approach distance: {d_min:.6f} km")

    fig, ax = plt.subplots(figsize=(8, 8))
    x, y = positions[:, 0], positions[:, 1]
    sc = ax.scatter(x, y, c=np.arange(len(x)), cmap="viridis", s=8, label="Orbit path")
    ax.plot([positions[0, 0]], [positions[0, 1]], "ro", label="Start point")
    ax.plot([x[i_star]], [y[i_star]], "kx", markersize=10, label="Closest approach")
    ax.set_xlabel("X [km]")
    ax.set_ylabel("Y [km]")
    ax.set_title("RK4 Orbit with J2")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, label="Time step index")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
