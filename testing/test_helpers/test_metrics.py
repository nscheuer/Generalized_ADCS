"""Unit tests for ADCS.helpers.metrics.

These pin the paper-table quantities and, critically, assert that the
metrics' attitude math is byte-identical to the figure-generating helper
``plot_controller_mc._rot_mat_vec`` so tables and plots can never diverge.
"""

import numpy as np
import pytest

from ADCS.helpers import metrics as M


def _quat_random(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_rotation_matches_plotting_helper():
    """quat_to_R_b2i must equal plot_controller_mc._rot_mat_vec exactly."""
    from ADCS.helpers.plotting_mc.plot_controller_mc import _rot_mat_vec

    q = _quat_random(64, seed=1)
    np.testing.assert_allclose(M.quat_to_R_b2i(q), _rot_mat_vec(q), rtol=0, atol=0)


def _state_with_quat(q):
    n = q.shape[0]
    s = np.zeros((n, 8))  # w(3) q(4) h(1)
    s[:, 3:7] = q
    return s


def test_boresight_error_aligned_and_orthogonal():
    n = 10
    q_id = np.tile([1.0, 0, 0, 0], (n, 1))  # identity -> body z stays ECI z
    state = _state_with_quat(q_id)
    goal_z = np.tile([0.0, 0, 1], (n, 1))
    err = M.boresight_error_deg(state, goal_z)
    np.testing.assert_allclose(err, 0.0, atol=1e-9)

    goal_x = np.tile([1.0, 0, 0], (n, 1))
    np.testing.assert_allclose(M.boresight_error_deg(state, goal_x), 90.0, atol=1e-9)


def test_goal_shape_4_equiv_3():
    q = _quat_random(20, seed=2)
    state = _state_with_quat(q)
    g3 = _quat_random(20, seed=3)[:, :3]
    g4 = np.column_stack([np.zeros(20), g3])  # width-4 -> cols 1:4 used
    np.testing.assert_allclose(
        M.boresight_error_deg(state, g3), M.boresight_error_deg(state, g4), atol=1e-12
    )


def test_settling_time_and_steady_state():
    t = np.arange(0.0, 100.0, 1.0)
    # Error 50 deg until t=40, then 1 deg thereafter.
    err = np.where(t < 40, 50.0, 1.0)
    assert M.settling_time(t, err, threshold_deg=5.0) == pytest.approx(40.0)
    # already-settled and never-settled edge cases
    assert M.settling_time(t, np.full_like(t, 0.5), 5.0) == pytest.approx(t[0])
    assert np.isnan(M.settling_time(t, np.full_like(t, 9.0), 5.0))
    # steady-state = mean over last 10%
    assert M.steady_state_error_deg(t, err, last_frac=0.1) == pytest.approx(1.0)


def _run(q, goal, t):
    return {"time": t, "state": _state_with_quat(q), "boresight_goal": goal}


def test_convergence_stats():
    n = 50
    t = np.arange(n, dtype=float)
    q_id = np.tile([1.0, 0, 0, 0], (n, 1))
    # run A: perfectly on target (0 deg) -> converged
    runA = _run(q_id, np.tile([0.0, 0, 1], (n, 1)), t)
    # run B: 90 deg off the whole time -> not converged, never settles
    runB = _run(q_id, np.tile([1.0, 0, 0], (n, 1)), t)
    s = M.convergence_stats([runA, runB], threshold_deg=5.0)
    assert s["n"] == 2
    assert s["n_converged"] == 1
    assert s["pct_converged"] == pytest.approx(50.0)
    assert s["mean_final"] == pytest.approx(45.0)
    assert s["mean_settle"] == pytest.approx(0.0)  # only runA settles, at t[0]=0


def test_allocation_metrics():
    d = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(M.allocation_direction_error_deg(d, d), [0.0], atol=1e-9)
    perp = np.array([[0.0, 2.0, 0.0]])
    np.testing.assert_allclose(M.allocation_direction_error_deg(perp, d), [90.0], atol=1e-9)
    assert np.isnan(M.allocation_direction_error_deg(np.zeros((1, 3)), d))[0]

    np.testing.assert_allclose(M.allocation_magnitude_ratio(d, d), [1.0], atol=1e-12)
    np.testing.assert_allclose(
        M.allocation_magnitude_ratio(0.5 * d, d), [0.5], atol=1e-12
    )
    np.testing.assert_allclose(
        M.allocation_magnitude_ratio(-d, d), [-1.0], atol=1e-12
    )  # QP can deliver anti-aligned torque


def test_table_emission(tmp_path):
    rows = [
        {"config": "3MTQ+1RW", "pct": 94.0, "mean_final": 1.2},
        {"config": "3MTQ+0RW", "pct": 84.0, "mean_final": 3.8},
    ]
    latex = M.metrics_table(rows, fmt="latex")
    assert r"\begin{tabular}" in latex and r"3MTQ\+1RW".replace("\\+", "+") in latex
    csv = M.metrics_table(rows, fmt="csv")
    assert csv.splitlines()[0] == "config,pct,mean_final"
    written = M.write_table(rows, str(tmp_path / "tab_mc"))
    assert sorted(p.split(".")[-1] for p in written) == ["csv", "md", "tex"]
    for p in written:
        assert open(p).read()


def test_simulation_results_adapter():
    """Paper 2 RunResults adapter must reuse the verified pointing math and
    expose RW momentum from the state vector."""
    from types import SimpleNamespace

    n = 30
    t = np.arange(n, dtype=float)
    q_id = np.tile([1.0, 0, 0, 0], (n, 1))
    state = np.zeros((n, 8))          # w(3) q(4) h(1)
    state[:, 3:7] = q_id
    state[:, 7] = np.linspace(0.01, 0.0, n)  # wheel desaturating to zero
    target = np.tile([0.0, 0, 1], (n, 1))

    run = SimpleNamespace(time_s=t, state_hist=state, target_hist=target,
                          control_hist=np.zeros((n, 4)))
    sim = SimpleNamespace(runs=[run, run])

    dicts = M.from_simulation_results(sim)
    assert len(dicts) == 2
    # identity attitude pointing at +z target -> ~0 deg, via the SAME core
    _, err = M.run_pointing_error(dicts[0])
    np.testing.assert_allclose(err, 0.0, atol=1e-9)
    assert M.converged(dicts[0], threshold_deg=5.0)
    # RW momentum extracted from state[:, 7:] and trends to zero (IV-A)
    h = M.rw_momentum(dicts[0])
    assert h.shape == (n, 1)
    assert h[0, 0] == pytest.approx(0.01) and h[-1, 0] == pytest.approx(0.0)
    # plain iterable of RunResults (no .runs) also accepted
    assert len(M.from_simulation_results([run])) == 1
