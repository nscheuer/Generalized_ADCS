"""Tests for covariance ownership, representations, and estimator operations."""

import numpy as np
import pytest

from ADCS import Covariance, EstimatorState
from ADCS.satellite_hardware.actuators import Actuator, RW
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import EstimatedSatellite
from ADCS.satellite_hardware.sensors import Sensor


SPD = np.array([[2.0, 0.4], [0.4, 1.0]])


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_representation_roundtrip_and_owned_outputs(form):
    source = SPD.copy()
    covariance = Covariance(source, form=form, coordinates="state_tangent")
    source[0, 0] = 99.0
    matrix = covariance.as_matrix()
    matrix[0, 0] = 88.0

    assert covariance.form == form
    assert covariance.coordinates == "state_tangent"
    np.testing.assert_allclose(covariance.as_matrix(), SPD)
    np.testing.assert_allclose(
        covariance.upper_factor().T @ covariance.upper_factor(), SPD
    )
    assert np.allclose(covariance.upper_factor(), np.triu(covariance.upper_factor()))


def test_singular_psd_has_valid_upper_factor():
    covariance = Covariance([[1.0, 1.0], [1.0, 1.0]], form="sqrt")
    factor = covariance.upper_factor()

    assert np.allclose(factor, np.triu(factor))
    np.testing.assert_allclose(factor.T @ factor, covariance.as_matrix(), atol=1e-12)


@pytest.mark.parametrize(
    ("matrix", "message"),
    [
        ([[1.0, 2.0], [0.0, 1.0]], "symmetric"),
        ([[1.0, 0.0], [0.0, -1.0]], "positive semidefinite"),
        ([[1.0, np.nan], [np.nan, 1.0]], "finite"),
    ],
)
def test_strict_validation_rejects_invalid_covariance(matrix, message):
    with pytest.raises(ValueError, match=message):
        Covariance(matrix)


def test_project_policy_repairs_small_negative_eigenvalue():
    covariance = Covariance(
        [[1.0, 0.0], [0.0, -1e-8]], form="sqrt", psd_policy="project"
    )

    assert np.linalg.eigvalsh(covariance.as_matrix()).min() >= -1e-14


def test_allow_indefinite_skips_eigen_validation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("eigendecomposition should not run")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    monkeypatch.setattr(np.linalg, "eigvalsh", fail)

    covariance = Covariance(
        [[1.0, 2.0], [2.0, 1.0]],
        psd_policy="allow_indefinite",
    )

    np.testing.assert_allclose(covariance.as_matrix(), [[1.0, 2.0], [2.0, 1.0]])


def test_copy_preserves_data_without_revalidation(monkeypatch):
    covariance = Covariance(SPD)

    def fail(*args, **kwargs):
        raise AssertionError("copy should trust covariance invariants")

    monkeypatch.setattr(np.linalg, "eigh", fail)
    monkeypatch.setattr(np.linalg, "eigvalsh", fail)

    copied = covariance.copy()

    assert copied is not covariance
    np.testing.assert_allclose(copied.as_matrix(), SPD)


def test_assign_is_atomic_and_retains_representation():
    covariance = Covariance(SPD, form="sqrt")
    before = covariance.as_matrix()

    with pytest.raises(ValueError):
        covariance.assign([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="retain shape"):
        covariance.assign(np.eye(3))

    assert covariance.form == "sqrt"
    np.testing.assert_allclose(covariance.as_matrix(), before)
    covariance.assign(np.eye(2) * 3.0)
    np.testing.assert_allclose(covariance.as_matrix(), np.eye(2) * 3.0)


def test_block_diagonal_accepts_empty_and_zero_size_blocks():
    covariance = Covariance.block_diagonal([np.zeros((0, 0)), SPD], form="sqrt")

    assert covariance.shape == SPD.shape
    np.testing.assert_allclose(covariance.as_matrix(), SPD)

    empty = Covariance.block_diagonal([], form="sqrt")
    assert empty.shape == (0, 0)
    assert empty.upper_factor().shape == (0, 0)


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_sigma_offsets_reconstruct_covariance(form):
    covariance = Covariance(SPD, form=form)
    scale = 1.7
    offsets = covariance.sigma_offsets(scale)

    assert offsets.shape == (4, 2)
    np.testing.assert_allclose(offsets[:2], -offsets[2:])
    np.testing.assert_allclose(offsets[:2].T @ offsets[:2] / scale**2, SPD)


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_block_subset_replace_and_zero_cross(form):
    covariance = Covariance.block_diagonal(
        [SPD, np.array([[3.0]])], form=form, coordinates="measurement"
    )
    np.testing.assert_allclose(covariance.subset([0, 2]).as_matrix(), np.diag([2.0, 3.0]))

    covariance.replace_block(slice(0, 2), np.eye(2) * 4.0)
    matrix = covariance.as_matrix()
    matrix[0, 2] = matrix[2, 0] = 0.5
    covariance.assign(matrix)
    covariance.zero_cross(slice(0, 2), [2])

    np.testing.assert_allclose(covariance.as_matrix(), np.diag([4.0, 4.0, 3.0]))


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_linear_prediction_and_joseph_update(form):
    covariance = Covariance(SPD, form=form, coordinates="state_tangent")
    transition = np.array([[1.0, 0.2], [0.0, 1.0]])
    process_noise = Covariance(np.diag([0.1, 0.2]))
    predicted = covariance.predicted_linear(transition, process_noise)
    expected_prediction = transition @ SPD @ transition.T + process_noise.as_matrix()
    np.testing.assert_allclose(predicted.as_matrix(), expected_prediction)

    h = np.array([[1.0, -0.5]])
    r = Covariance([[0.3]], coordinates="measurement")
    gain, posterior = predicted.updated_linear(h, r)
    innovation = h @ expected_prediction @ h.T + r.as_matrix()
    expected_gain = expected_prediction @ h.T @ np.linalg.inv(innovation)
    residual = np.eye(2) - expected_gain @ h
    expected_posterior = (
        residual @ expected_prediction @ residual.T
        + expected_gain @ r.as_matrix() @ expected_gain.T
    )

    np.testing.assert_allclose(gain, expected_gain)
    np.testing.assert_allclose(posterior.as_matrix(), expected_posterior)
    assert posterior.form == form


@pytest.mark.parametrize("weight", [0.25, -0.25])
def test_sqrt_rank_updated_uses_weighted_cholesky_update(weight):
    vector = np.array([0.5, -0.25])
    covariance = Covariance(SPD, form="sqrt")

    updated = covariance.rank_updated(vector, weight=weight)

    expected = SPD + weight * np.outer(vector, vector)
    np.testing.assert_allclose(updated.as_matrix(), expected, atol=1e-12)
    assert updated.form == "sqrt"


def test_sqrt_rank_update_from_singular_factor_keeps_qr_fallback():
    covariance = Covariance.zeros(2, form="sqrt")

    updated = covariance.rank_updated(np.eye(2), weight=2.0)

    np.testing.assert_allclose(updated.as_matrix(), np.eye(2) * 2.0)


def test_weighted_deviations_with_negative_weight_stays_in_sqrt_form():
    deviations = np.array(
        [
            [0.5, -0.25],
            [0.0, 0.75],
            [0.25, 0.5],
        ]
    )
    weights = np.array([-0.2, 0.4, 0.3])
    noise = Covariance(np.eye(2) * 1.5)

    covariance = Covariance.from_weighted_deviations(
        deviations,
        weights,
        noise,
        form="sqrt",
    )

    expected = np.einsum("i,ij,ik->jk", weights, deviations, deviations)
    expected += noise.as_matrix()
    np.testing.assert_allclose(covariance.as_matrix(), expected, atol=1e-12)
    assert covariance.form == "sqrt"


@pytest.mark.parametrize("form", ["full", "sqrt"])
def test_unscented_prediction_and_update(form):
    state_deviations = np.array(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    weights = np.full(4, 0.25)
    process_noise = Covariance(np.diag([0.1, 0.2]))
    prior = Covariance.identity(2, form=form, coordinates="state_tangent")
    predicted = prior.predicted_unscented(state_deviations, weights, process_noise)
    expected = np.einsum("i,ij,ik->jk", weights, state_deviations, state_deviations)
    expected += process_noise.as_matrix()
    np.testing.assert_allclose(predicted.as_matrix(), expected)

    measurement_deviations = state_deviations @ np.array([[1.0], [0.5]])
    measurement_noise = Covariance([[0.4]], coordinates="measurement")
    gain, posterior = predicted.updated_unscented(
        state_deviations, measurement_deviations, weights, measurement_noise
    )
    cross = Covariance.cross_covariance(
        state_deviations, measurement_deviations, weights
    )
    innovation = np.einsum(
        "i,ij,ik->jk",
        weights,
        measurement_deviations,
        measurement_deviations,
    )
    innovation += measurement_noise.as_matrix()

    np.testing.assert_allclose(gain, cross @ np.linalg.inv(innovation))
    np.testing.assert_allclose(
        posterior.as_matrix(), predicted.as_matrix() - gain @ innovation @ gain.T
    )


def test_estimator_state_owns_covariance_and_preserves_legacy_matrix_api():
    state_covariance = Covariance.identity(6, scale=2.0, form="sqrt")
    process_noise = Covariance.identity(6, scale=0.1, form="sqrt")
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        covariance=state_covariance,
        process_noise=process_noise,
    )

    assert state.covariance.form == "sqrt"
    assert state.process_noise.form == "sqrt"
    assert state.covariance is not state_covariance
    np.testing.assert_allclose(state.cov, np.eye(6) * 2.0)

    state.cov = np.eye(6) * 3.0
    assert state.covariance.form == "sqrt"
    np.testing.assert_allclose(state.covariance.as_matrix(), np.eye(6) * 3.0)


def test_estimator_state_legacy_matrix_path_preserves_indefinite_filter_fixture():
    state = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0])
    legacy_matrix = np.eye(6)
    legacy_matrix[0, 1] = legacy_matrix[1, 0] = 2.0

    state.cov = legacy_matrix

    np.testing.assert_array_equal(state.cov, legacy_matrix)
    with pytest.raises(ValueError, match="positive semidefinite"):
        Covariance(legacy_matrix)


def test_estimator_state_normalizes_covariance_policy():
    strict_covariance = Covariance.identity(6, scale=1.0, psd_policy="strict")
    state = EstimatorState(
        w=np.zeros(3),
        q=[1.0, 0.0, 0.0, 0.0],
        covariance=strict_covariance,
    )
    legacy_matrix = np.eye(6)
    legacy_matrix[0, 1] = legacy_matrix[1, 0] = 2.0

    assert state.covariance.psd_policy == "allow_indefinite"
    state.cov = legacy_matrix
    np.testing.assert_allclose(state.cov, legacy_matrix)


def test_estimator_state_interpolate_preserves_allow_indefinite_policy():
    legacy_matrix = np.eye(6)
    legacy_matrix[0, 1] = legacy_matrix[1, 0] = 2.0
    first = EstimatorState(w=np.zeros(3), q=[1.0, 0.0, 0.0, 0.0], cov=legacy_matrix)
    second = EstimatorState(w=np.ones(3), q=[1.0, 0.0, 0.0, 0.0], cov=legacy_matrix)

    midpoint = first.interpolate(second, 0.5)

    assert midpoint.covariance.psd_policy == "allow_indefinite"
    np.testing.assert_allclose(midpoint.cov, legacy_matrix)


def test_noise_bias_sensor_and_actuator_covariance_ownership():
    noise = Noise(std_noise=[0.2, 0.3])
    sensor = Sensor(output_length=2, noise=noise)
    actuator = Actuator(
        axis=np.array([1.0, 0.0, 0.0]),
        u_max=1.0,
        noise=Noise(std_noise=0.4),
        bias=Bias(std_bias=0.05),
    )

    np.testing.assert_allclose(
        sensor.measurement_covariance(form="sqrt").as_matrix(), np.diag([0.04, 0.09])
    )
    np.testing.assert_allclose(actuator.control_covariance().as_matrix(), [[0.16]])
    np.testing.assert_allclose(
        actuator.bias_process_covariance(4.0).as_matrix(), [[0.01]]
    )


def test_estimated_satellite_aggregates_owner_covariances():
    sensors = [
        Sensor(output_length=1, noise=Noise(std_noise=0.2)),
        Sensor(output_length=2, noise=Noise(std_noise=[0.3, 0.4])),
    ]
    actuator = Actuator(
        axis=np.array([1.0, 0.0, 0.0]), u_max=1.0, noise=Noise(std_noise=0.5)
    )
    wheel = RW(
        axis=np.array([0.0, 1.0, 0.0]),
        max_torque=0.1,
        J=0.01,
        h=0.0,
        h_max=1.0,
        h_meas_noise=Noise(std_noise=0.6),
    )
    satellite = EstimatedSatellite(sensors=sensors, actuators=[actuator, wheel])

    np.testing.assert_allclose(
        satellite.control_covariance(form="sqrt").as_matrix(), np.diag([0.25, 0.0])
    )
    np.testing.assert_allclose(
        satellite.measurement_covariance([True, False]).as_matrix(),
        np.diag([0.04, 0.36]),
    )
    np.testing.assert_allclose(
        satellite.measurement_covariance([True, True]).as_matrix(),
        satellite.sensor_cov([True, True]),
    )
