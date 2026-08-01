import numpy as np

from ADCS.controller.helpers.trajectory import Trajectory
from ADCS.state import State


def _make_traj(states, times=None):
    n = len(states)
    times = np.arange(n, dtype=float) if times is None else np.asarray(times, dtype=float)
    u = np.zeros((n - 1, 3))
    K = np.zeros((n - 1, 3, 6))
    S = np.zeros((n, 6, 6))
    return Trajectory(times, states, u, K, S)


def test_get_state_at_interpolates_shortest_arc_across_sign_flip():
    # Solvers may return q at one knot and -q at the next; sign-blind lerp of
    # antipodal representations collapses toward the origin and renormalizes
    # garbage. The midpoint must stay on the geodesic between the rotations.
    theta = 0.3
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    q1 = np.array([np.cos(theta / 2), np.sin(theta / 2), 0.0, 0.0])
    traj = _make_traj([State(w=np.zeros(3), q=q0), State(w=np.zeros(3), q=-q1)])

    mid = traj.get_state_at(0.5)

    expected = np.array([np.cos(theta / 4), np.sin(theta / 4), 0.0, 0.0])
    assert np.isclose(np.linalg.norm(mid.q), 1.0)
    assert min(np.linalg.norm(mid.q - expected), np.linalg.norm(mid.q + expected)) < 1e-4


def test_get_state_at_endpoints_and_linear_blocks():
    s0 = State(w=[1.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.1])
    s1 = State(w=[3.0, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0], h=[0.5])
    traj = _make_traj([s0, s1])

    assert np.allclose(traj.get_state_at(0.0).as_array(), s0.as_array())
    mid = traj.get_state_at(0.5)
    assert np.allclose(mid.w, [2.0, 0.0, 0.0])
    assert np.allclose(mid.h, [0.3])


def test_get_state_at_copies_do_not_alias_internal_states():
    s0 = State(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.1])
    s1 = State(w=np.ones(3), q=[1.0, 0.0, 0.0, 0.0], h=[0.2])
    traj = _make_traj([s0, s1])

    s0.w[:] = 99.0  # mutate the input after construction
    assert np.allclose(traj.get_state_at(0.0).w, np.zeros(3))

    out = traj.get_state_at(0.0)
    out.w[:] = -5.0  # mutate the returned state
    assert np.allclose(traj.get_state_at(0.0).w, np.zeros(3))
