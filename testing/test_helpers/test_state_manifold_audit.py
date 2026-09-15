"""Manifold-layer regressions from the September 2026 estimator audit.

The core algebra (tangent maps, retraction Jacobians, transport, layout) was
verified correct to 1e-11 or better; these tests pin the defects the audit
found in the less-travelled branches, plus two properties worth guarding.
"""

from __future__ import annotations

import numpy as np
import pytest

from ADCS.state import EstimatorState, State, _rotation_vector_reset_jacobian


REDUCED_CHARTS = ["quaternion_vector", "rotation_vector", "mrp", "two_mrp", "cayley"]


def _state(q) -> State:
    return State(w=np.zeros(3), q=np.asarray(q, float))


def _attitude(v) -> np.ndarray:
    return np.concatenate((np.zeros(3), np.asarray(v, float)))


IDENTITY = _state([1.0, 0.0, 0.0, 0.0])


# --- State.mean --------------------------------------------------------------

def test_state_mean_converges_in_full_quaternion_mode():
    """Used to raise RuntimeError on every input: the raw step keeps a radial
    part that the normalizing retraction discards, so it never shrank."""
    other = IDENTITY.plus(_attitude([0.0, 1.0e-3, 0.0, 0.0]), quaternion_mode="full_quaternion")
    mean = State.mean([IDENTITY, other], quaternion_mode="full_quaternion")
    assert np.isclose(np.linalg.norm(mean.q), 1.0)
    halfway = IDENTITY.plus(_attitude([0.0, 5.0e-4, 0.0, 0.0]), quaternion_mode="full_quaternion")
    assert np.allclose(mean.q, halfway.q, atol=1.0e-8)


@pytest.mark.parametrize("mode", REDUCED_CHARTS)
def test_state_mean_fixed_point_defect_in_reduced_charts(mode):
    rng = np.random.default_rng(1)
    points = [IDENTITY.plus(_attitude(0.3 * rng.standard_normal(3)), quaternion_mode=mode)
              for _ in range(9)]
    mean = State.mean(points, quaternion_mode=mode)
    defect = sum(p.minus(mean, quaternion_mode=mode) for p in points) / len(points)
    assert np.linalg.norm(defect) < 1.0e-12


def test_state_mean_rejects_a_nonpositive_tolerance_up_front():
    with pytest.raises(ValueError, match="tolerance"):
        State.mean([IDENTITY, IDENTITY], tolerance=0.0)
    with pytest.raises(TypeError, match="reference"):
        State.mean([IDENTITY, IDENTITY], reference=1.0)


@pytest.mark.parametrize("mode", REDUCED_CHARTS)
def test_state_mean_of_an_antipodal_pair_fails_loudly_or_succeeds(mode):
    """Two attitudes 180 degrees apart. Every chart must either return a mean or
    raise a typed error; a bare ZeroDivisionError from numba is neither."""
    far = _state([0.0, 1.0, 0.0, 0.0])
    if mode == "cayley":
        with pytest.raises(ValueError, match="singular"):
            State.mean([IDENTITY, far], quaternion_mode=mode)
    else:
        mean = State.mean([IDENTITY, far], quaternion_mode=mode)
        assert np.isclose(np.linalg.norm(mean.q), 1.0)


# --- two_mrp chart ----------------------------------------------------------

def test_two_mrp_plus_is_continuous_across_180_degrees():
    """The helper modes used into and out of the chart disagreed on the sign of
    the scalar part, so the stored quaternion flipped at |delta| = 2."""
    below = IDENTITY.plus(_attitude([2.0 - 1.0e-9, 0.0, 0.0]), quaternion_mode="two_mrp")
    above = IDENTITY.plus(_attitude([2.0 + 1.0e-9, 0.0, 0.0]), quaternion_mode="two_mrp")
    assert np.allclose(below.q, above.q, atol=1.0e-6)


@pytest.mark.parametrize("mode", ["mrp", "two_mrp"])
@pytest.mark.parametrize("degrees", [200.0, 270.0, 350.0])
def test_mrp_family_round_trips_past_180_degrees_with_shortest_false(mode, degrees):
    scale = 2.0 if mode == "two_mrp" else 1.0
    delta = _attitude([scale * np.tan(np.deg2rad(degrees) / 4.0), 0.0, 0.0])
    back = IDENTITY.plus(delta, quaternion_mode=mode).minus(IDENTITY, quaternion_mode=mode, shortest=False)
    np.testing.assert_allclose(back[3:], delta[3:], rtol=0.0, atol=1.0e-10)


def test_two_mrp_retraction_jacobian_is_well_conditioned_at_the_boundary():
    jacobian = IDENTITY.retraction_jacobian(_attitude([2.0, 0.0, 0.0]), quaternion_mode="two_mrp")
    assert np.linalg.norm(jacobian) < 10.0  # was 8e10 across the sign discontinuity


# --- chart singularities raise typed errors ---------------------------------

def test_cayley_minus_at_180_degrees_raises_a_typed_error():
    far = _state([0.0, 1.0, 0.0, 0.0])
    for shortest in (True, False):
        with pytest.raises(ValueError, match="singular"):
            far.minus(IDENTITY, quaternion_mode="cayley", shortest=shortest)


@pytest.mark.parametrize("mode", ["mrp", "two_mrp"])
def test_mrp_family_minus_at_scalar_minus_one_raises_a_typed_error(mode):
    flipped = _state([-1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="singular"):
        flipped.minus(IDENTITY, quaternion_mode=mode, shortest=False)
    # shortest=True resolves the sign and must keep working
    np.testing.assert_allclose(flipped.minus(IDENTITY, quaternion_mode=mode), np.zeros(6), atol=1.0e-15)


def test_full_quaternion_delta_to_vector_names_the_right_problem():
    with pytest.raises(ValueError, match="full_quaternion"):
        State.quaternion_delta_to_vector(np.zeros(4), mode="full_quaternion")


# --- rotation-vector reset Jacobian near zero -------------------------------

@pytest.mark.parametrize("theta", [1.0e-9, 1.0e-8, 3.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 5.0e-4])
def test_rotation_vector_reset_jacobian_small_angle_accuracy(theta):
    """The closed forms cancel catastrophically below ~1e-5 (both coefficients
    were exactly zero at 1e-8); compare against a four-term series."""
    vector = np.array([0.0, 0.0, theta])
    a = 0.5 - theta**2 / 24.0 + theta**4 / 720.0 - theta**6 / 40320.0
    b = 1.0 / 6.0 - theta**2 / 120.0 + theta**4 / 5040.0 - theta**6 / 362880.0
    cross = np.array([[0.0, -theta, 0.0], [theta, 0.0, 0.0], [0.0, 0.0, 0.0]])
    reference = np.eye(3) - a * cross + b * (cross @ cross)
    assert np.abs(_rotation_vector_reset_jacobian(vector, "right") - reference).max() < 1.0e-14


# --- properties worth pinning -------------------------------------------------

@pytest.mark.parametrize("mode", ["quaternion_vector", "cayley"])
@pytest.mark.parametrize("degrees", [90.0, 150.0, 179.0, 179.9])
def test_retraction_jacobian_condition_number_is_the_chart_geometry(mode, degrees):
    """The growth toward 180 degrees is exactly sec(theta/2), a property of the
    chart and not an implementation defect. Pin it so it is never 'fixed'."""
    theta = np.deg2rad(degrees)
    if mode == "quaternion_vector":
        magnitude = 2.0 * np.sin(theta / 2.0)
    else:
        magnitude = np.tan(theta / 2.0)
    jacobian = IDENTITY.retraction_jacobian(_attitude([magnitude, 0.0, 0.0]), quaternion_mode=mode)
    attitude = jacobian[3:6, 3:6]
    assert np.linalg.cond(attitude) == pytest.approx(1.0 / np.cos(theta / 2.0), rel=1.0e-5)


def test_cov_property_is_an_owned_copy_and_the_setter_writes():
    state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=np.eye(6), int_cov=np.zeros((6, 6)))
    state.cov[0, 0] = 99.0  # silently discarded: the property returns a copy
    assert state.cov[0, 0] == 1.0
    matrix = state.cov
    matrix[0, 0] = 99.0
    state.cov = matrix
    assert state.cov[0, 0] == 99.0
