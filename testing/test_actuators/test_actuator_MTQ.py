from dataclasses import dataclass

import numdifftools as nd
import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.helpers.math_helpers import rot_mat
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import MTQ
from ADCS.satellite_hardware.errors import Bias, Noise
from ADCS.satellite_hardware.satellite import Satellite
from ADCS.state import State


@dataclass(frozen=True)
class MTQCase:
    axis_raw: np.ndarray
    max_torque: float
    u: float
    w: np.ndarray
    q: np.ndarray
    x: State
    b_eci: np.ndarray
    orbital_state: Orbital_State

    @property
    def axis(self) -> np.ndarray:
        return self.axis_raw / np.linalg.norm(self.axis_raw)

    @property
    def body_field(self) -> np.ndarray:
        return self.orbital_state.get_state_vector(x=self.x)["b"]

    @property
    def body_field_jacobian(self) -> np.ndarray:
        return self.orbital_state.get_state_vector(x=self.x)["db"]

    @property
    def body_field_hessian(self) -> np.ndarray:
        return self.orbital_state.get_state_vector(x=self.x)["ddb"]


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _make_case(seed: int = 0) -> MTQCase:
    rng = np.random.default_rng(seed)
    axis_raw = 3.0 * _unit(rng.normal(size=3))
    q = _unit(rng.normal(size=4))
    w = 0.05 * _unit(rng.normal(size=3))
    x = State(w=w, q=q)
    b_eci = 1e-5 * _unit(rng.normal(size=3))
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=b_eci,
    )
    return MTQCase(
        axis_raw=axis_raw,
        max_torque=10.0,
        u=float(rng.normal()),
        w=w,
        q=q,
        x=x,
        b_eci=b_eci,
        orbital_state=orbital_state,
    )


def _make_mtq(case: MTQCase, *, bias: Bias | None = None, noise: Noise | None = None) -> MTQ:
    return MTQ(axis=case.axis_raw, max_torque=case.max_torque, bias=bias, noise=noise)


def _torque_scale(case: MTQCase) -> np.ndarray:
    return -np.cross(case.body_field, case.axis)


def _expected_state_jacobian(case: MTQCase, commanded_value: float) -> np.ndarray:
    return np.vstack(
        [np.zeros((3, 3)), -np.cross(case.body_field_jacobian, case.axis) * commanded_value]
    )


def _expected_state_hessian(case: MTQCase, commanded_value: float) -> np.ndarray:
    expected = np.zeros((7, 7, 3))
    expected[3:7, 3:7, :] = -np.cross(case.body_field_hessian, case.axis) * commanded_value
    return expected


def _assert_zero_storage_api(mtq: MTQ, case: MTQCase, *, has_bias: bool) -> None:
    expected_dbias_shape = (1, 0) if has_bias else (0, 0)
    expected_dudbias_shape = (1, 1, 0) if has_bias else (1, 0, 0)
    expected_dbiasdbias_shape = (1, 1, 0) if has_bias else (0, 0, 0)
    expected_dbiasdbasestate_shape = (1, 7, 0) if has_bias else (0, 7, 0)
    expected_dbiasdh_shape = (1, 0, 0) if has_bias else (0, 0, 0)

    assert mtq.storage_torque(u=case.u, x=case.x, os=case.orbital_state).shape == (0,)
    np.testing.assert_allclose(mtq.storage_torque(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0,)))

    np.testing.assert_allclose(mtq.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0)))
    assert mtq.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state).shape == (1, 0)

    np.testing.assert_allclose(
        mtq.dstor_torq__dbias(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros(expected_dbias_shape),
    )
    assert mtq.dstor_torq__dbias(u=case.u, x=case.x, os=case.orbital_state).shape == expected_dbias_shape

    np.testing.assert_allclose(
        mtq.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros((7, 0)),
    )
    assert mtq.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state).shape == (7, 0)

    np.testing.assert_allclose(mtq.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0)))
    assert mtq.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state).shape == (0, 0)

    np.testing.assert_allclose(
        mtq.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros((1, 1, 0)),
    )
    assert mtq.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state).shape == (1, 1, 0)

    np.testing.assert_allclose(
        mtq.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros(expected_dudbias_shape),
    )
    assert mtq.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state).shape == expected_dudbias_shape

    np.testing.assert_allclose(
        mtq.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros((1, 7, 0)),
    )
    assert mtq.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state).shape == (1, 7, 0)

    np.testing.assert_allclose(mtq.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0, 0)))
    assert mtq.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state).shape == (1, 0, 0)

    np.testing.assert_allclose(
        mtq.ddstor_torq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros(expected_dbiasdbias_shape),
    )
    assert mtq.ddstor_torq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state).shape == expected_dbiasdbias_shape

    np.testing.assert_allclose(
        mtq.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros(expected_dbiasdbasestate_shape),
    )
    assert mtq.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state).shape == expected_dbiasdbasestate_shape

    np.testing.assert_allclose(
        mtq.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros(expected_dbiasdh_shape),
    )
    assert mtq.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state).shape == expected_dbiasdh_shape

    np.testing.assert_allclose(
        mtq.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros((7, 7, 0)),
    )
    assert mtq.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state).shape == (7, 7, 0)

    np.testing.assert_allclose(
        mtq.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state),
        np.zeros((7, 0, 0)),
    )
    assert mtq.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state).shape == (7, 0, 0)

    np.testing.assert_allclose(mtq.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0, 0)))
    assert mtq.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state).shape == (0, 0, 0)


def _projected_hessian_clean(mtq: MTQ, case: MTQCase, direction: np.ndarray) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        x = State.from_array(values[1:])
        local = _make_mtq(case)
        return float(np.dot(local.torque(u=u, x=x, os=case.orbital_state), direction))

    point = np.concatenate([[case.u], case.x.as_array()])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def _projected_hessian_biased(mtq: MTQ, case: MTQCase, direction: np.ndarray, bias_value: float) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        b = values[1]
        x = State.from_array(values[2:])
        local = _make_mtq(case, bias=Bias(bias=b, std_bias=0.0))
        return float(np.dot(local.torque(u=u, x=x, os=case.orbital_state), direction))

    point = np.concatenate([[case.u], [bias_value], case.x.as_array()])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def _expected_seeded_torques(
    case: MTQCase,
    *,
    times: list[float],
    bias0: float = 0.0,
    std_bias: float = 0.0,
    noise0: float = 0.0,
    std_noise: float = 0.0,
) -> list[np.ndarray]:
    expected = []
    bias_curr = float(bias0)
    noise_curr = float(noise0)
    bias_active = not (bias0 == 0.0 and std_bias == 0.0)
    noise_active = not (noise0 == 0.0 and std_noise == 0.0)
    last_bias_time = None

    for time in times:
        torque = _torque_scale(case) * (case.u + (bias_curr if bias_active else 0.0))
        if noise_active:
            torque = torque + noise_curr
        expected.append(torque)

        if last_bias_time is None:
            last_bias_time = time
        else:
            dt_sec = (time - last_bias_time) * TimeConstants.cent2sec
            if dt_sec > 0:
                bias_curr = np.random.normal(loc=bias_curr, scale=std_bias * np.sqrt(dt_sec))
                last_bias_time = time

        noise_curr = np.random.normal(loc=0.0, scale=std_noise)

    return expected


def test_mtq_setup_normalizes_axis_and_copies_error_models() -> None:
    case = _make_case(0)
    bias = Bias(bias=0.13, std_bias=0.03)
    noise = Noise(noise=0.0, std_noise=0.24)

    mtq = _make_mtq(case, bias=bias, noise=noise)

    np.testing.assert_allclose(mtq.axis, case.axis)
    assert mtq.u_max == case.max_torque
    assert mtq.bias is not bias
    assert mtq.noise is not noise
    np.testing.assert_allclose(mtq.bias.bias, bias.bias)
    np.testing.assert_allclose(mtq.bias.std_bias, bias.std_bias)
    np.testing.assert_allclose(mtq.noise.noise, noise.noise)
    np.testing.assert_allclose(mtq.noise.std_noise, noise.std_noise)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [
        (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1e-5])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.0, 1e-5, 0.0])),
    ],
)
def test_mtq_torque_axis_aligned_cases(axis: np.ndarray, expected: np.ndarray) -> None:
    mtq = MTQ(axis=axis, max_torque=1.0)
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=1e-5 * np.array([1.0, 0.0, 0.0]),
    )
    x = State(w=[0.01, 0.0, 0.0], q=[1.0, 0.0, 0.0, 0.0])

    np.testing.assert_allclose(mtq.torque(u=0.0, x=x, os=orbital_state), np.zeros(3))
    np.testing.assert_allclose(mtq.torque(u=1.0, x=x, os=orbital_state), expected)


def test_mtq_torque_matches_satellite_dynamics() -> None:
    case = _make_case(1)
    q_j = _unit(np.array([0.2, -0.1, 0.3, 0.9]))
    rmat_j = rot_mat(q_j)
    j_body = rmat_j @ np.diag([2.0, 3.0, 10.0]) @ rmat_j.T
    rmat = rot_mat(case.q)
    j_eci = rmat @ j_body @ rmat.T
    w_eci = rmat @ case.w
    h_eci = j_eci @ w_eci
    m_eci = rmat @ (case.axis * case.u)
    torque_eci = np.cross(case.b_eci, m_eci)

    expected_wdot = -rmat.T @ np.linalg.inv(j_eci) @ (np.cross(w_eci, h_eci) + torque_eci)
    expected_qdot = 0.5 * np.concatenate(
        [[-np.dot(case.q[1:], case.w)], case.q[0] * case.w + np.cross(case.q[1:], case.w)]
    )

    mtqs = [MTQ(axis=axis, max_torque=1.0) for axis in MathConstants.unitvecs]
    sat = Satellite(J_0=j_body, actuators=mtqs)
    dx = sat.dynamics_core(x=case.x, u=case.axis * case.u, orbital_state=case.orbital_state)

    np.testing.assert_allclose(dx, np.concatenate([expected_wdot, expected_qdot]))


def test_mtq_clean_torque_matches_closed_form() -> None:
    case = _make_case(2)
    mtq = _make_mtq(case)

    np.testing.assert_allclose(
        mtq.torque(u=case.u, x=case.x, os=case.orbital_state),
        _torque_scale(case) * case.u,
    )


def test_mtq_biased_torque_matches_closed_form() -> None:
    case = _make_case(3)
    bias = Bias(bias=0.17, std_bias=0.0)
    mtq = _make_mtq(case, bias=bias)

    np.testing.assert_allclose(
        mtq.torque(u=case.u, x=case.x, os=case.orbital_state),
        _torque_scale(case) * (case.u + bias.get_bias(case.orbital_state.J2000)),
    )


def test_mtq_clean_jacobians_match_closed_form_and_finite_difference() -> None:
    case = _make_case(4)
    mtq = _make_mtq(case)

    numeric_du = np.array(nd.Jacobian(lambda value: mtq.torque(u=value, x=case.x, os=case.orbital_state))(case.u)).T
    numeric_dx = np.array(
        nd.Jacobian(lambda value: mtq.torque(u=case.u, x=State.from_array(value), os=case.orbital_state))(case.x.as_array().tolist())
    ).T

    np.testing.assert_allclose(mtq.dtorq__du(u=case.u, x=case.x, os=case.orbital_state), _torque_scale(case).reshape(1, 3))
    np.testing.assert_allclose(mtq.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 3)))
    np.testing.assert_allclose(
        mtq.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state),
        _expected_state_jacobian(case, case.u),
    )
    np.testing.assert_allclose(mtq.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 3)))

    jac_du, jac_dx = mtq.jacobians(u=case.u, x=case.x, os=case.orbital_state)
    np.testing.assert_allclose(jac_du, mtq.dtorq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(jac_dx, mtq.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))

    np.testing.assert_allclose(numeric_du, mtq.dtorq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, mtq.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))


def test_mtq_biased_jacobians_match_closed_form_and_finite_difference() -> None:
    case = _make_case(5)
    bias = Bias(bias=0.19, std_bias=0.0)
    mtq = _make_mtq(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric_du = np.array(nd.Jacobian(lambda value: mtq.torque(u=value, x=case.x, os=case.orbital_state))(case.u)).T
    numeric_db = np.array(
        nd.Jacobian(
            lambda value: _make_mtq(case, bias=Bias(bias=value, std_bias=0.0)).torque(
                u=case.u, x=case.x, os=case.orbital_state
            )
        )(bias_value)
    ).T
    numeric_dx = np.array(
        nd.Jacobian(lambda value: mtq.torque(u=case.u, x=State.from_array(value), os=case.orbital_state))(case.x.as_array().tolist())
    ).T

    expected_scale = _torque_scale(case).reshape(1, 3)
    expected_dx = _expected_state_jacobian(case, case.u + bias_value)

    np.testing.assert_allclose(mtq.dtorq__du(u=case.u, x=case.x, os=case.orbital_state), expected_scale)
    np.testing.assert_allclose(mtq.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state), expected_scale)
    np.testing.assert_allclose(mtq.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dx)
    np.testing.assert_allclose(mtq.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 3)))

    np.testing.assert_allclose(numeric_du, mtq.dtorq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_db, mtq.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, mtq.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))


def test_mtq_clean_hessian_blocks_match_closed_form() -> None:
    case = _make_case(6)
    mtq = _make_mtq(case)

    expected_dudx = _expected_state_jacobian(case, 1.0).reshape(1, 7, 3)
    expected_dxdx = _expected_state_hessian(case, case.u)

    np.testing.assert_allclose(mtq.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dudx)
    np.testing.assert_allclose(mtq.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 7, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dxdx)
    np.testing.assert_allclose(mtq.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0, 3)))

    hess_dudx, hess_dxdx = mtq.hessians(u=case.u, x=case.x, os=case.orbital_state)
    np.testing.assert_allclose(hess_dudx, mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(hess_dxdx, mtq.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state))


def test_mtq_biased_hessian_blocks_match_closed_form() -> None:
    case = _make_case(7)
    bias = Bias(bias=0.23, std_bias=0.0)
    mtq = _make_mtq(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    expected_dudx = _expected_state_jacobian(case, 1.0).reshape(1, 7, 3)
    expected_dxdx = _expected_state_hessian(case, case.u + bias_value)

    np.testing.assert_allclose(mtq.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dudx)
    np.testing.assert_allclose(mtq.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dudx)
    np.testing.assert_allclose(mtq.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), expected_dxdx)
    np.testing.assert_allclose(mtq.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 0, 3)))
    np.testing.assert_allclose(mtq.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 0, 3)))


@pytest.mark.parametrize("direction", MathConstants.unitvecs)
def test_mtq_clean_projected_hessian_matches_finite_difference(direction: np.ndarray) -> None:
    case = _make_case(8)
    mtq = _make_mtq(case)

    numeric = _projected_hessian_clean(mtq, case, direction)
    expected = np.block(
        [
            [
                mtq.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                mtq.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-18)


@pytest.mark.parametrize("direction", MathConstants.unitvecs)
def test_mtq_biased_projected_hessian_matches_finite_difference(direction: np.ndarray) -> None:
    case = _make_case(9)
    bias = Bias(bias=0.11, std_bias=0.0)
    mtq = _make_mtq(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric = _projected_hessian_biased(mtq, case, direction, bias_value)
    expected = np.block(
        [
            [
                mtq.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                mtq.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (mtq.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                mtq.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                mtq.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (mtq.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                (mtq.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                mtq.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-18)


def test_mtq_clean_storage_api_is_zero() -> None:
    case = _make_case(10)
    _assert_zero_storage_api(_make_mtq(case), case, has_bias=False)


def test_mtq_biased_storage_api_is_zero() -> None:
    case = _make_case(11)
    _assert_zero_storage_api(_make_mtq(case, bias=Bias(bias=0.07, std_bias=0.0)), case, has_bias=True)


def test_mtq_bias_updates_follow_seeded_random_walk() -> None:
    case = _make_case(12)
    bias0 = 0.08
    std_bias = 0.03
    mtq = _make_mtq(case, bias=Bias(bias=bias0, std_bias=std_bias))

    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(123)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(mtq.torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(123)
    expected = _expected_seeded_torques(case, times=times, bias0=bias0, std_bias=std_bias)

    np.testing.assert_allclose(actual, expected)


def test_mtq_noise_updates_follow_seeded_samples() -> None:
    case = _make_case(13)
    std_noise = 0.24
    mtq = _make_mtq(case, noise=Noise(noise=0.0, std_noise=std_noise))

    np.random.seed(456)
    actual = [mtq.torque(u=case.u, x=case.x, os=case.orbital_state) for _ in range(4)]

    np.random.seed(456)
    expected = _expected_seeded_torques(case, times=[case.orbital_state.J2000] * 4, noise0=0.0, std_noise=std_noise)

    np.testing.assert_allclose(actual, expected)


def test_mtq_bias_and_noise_updates_follow_seeded_samples() -> None:
    case = _make_case(14)
    bias0 = 0.05
    std_bias = 0.03
    std_noise = 0.18
    mtq = _make_mtq(
        case,
        bias=Bias(bias=bias0, std_bias=std_bias),
        noise=Noise(noise=0.0, std_noise=std_noise),
    )

    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(789)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(mtq.torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(789)
    expected = _expected_seeded_torques(
        case,
        times=times,
        bias0=bias0,
        std_bias=std_bias,
        noise0=0.0,
        std_noise=std_noise,
    )

    np.testing.assert_allclose(actual, expected)
