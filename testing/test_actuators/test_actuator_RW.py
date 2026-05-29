from dataclasses import dataclass

import numdifftools as nd
import numpy as np
import pytest

from ADCS.helpers.math_constants import MathConstants
from ADCS.orbits.ephemeris import Ephemeris
from ADCS.orbits.orbital_state import Orbital_State
from ADCS.orbits.universal_constants import TimeConstants
from ADCS.satellite_hardware.actuators import RW
from ADCS.satellite_hardware.errors import Bias, Noise


@dataclass(frozen=True)
class RWCase:
    axis_raw: np.ndarray
    max_torque: float
    J: float
    h: float
    h_max: float
    u: float
    x: np.ndarray
    orbital_state: Orbital_State

    @property
    def axis(self) -> np.ndarray:
        return self.axis_raw / np.linalg.norm(self.axis_raw)


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _make_case(seed: int = 0) -> RWCase:
    rng = np.random.default_rng(seed)
    axis_raw = 3.0 * _unit(rng.normal(size=3))
    q = _unit(rng.normal(size=4))
    w = 0.05 * _unit(rng.normal(size=3))
    x = np.concatenate((w, q))
    orbital_state = Orbital_State(
        ephem=Ephemeris(),
        J2000=0.22,
        R=np.array([7000.0, 0.0, 0.0]),
        V=np.array([0.0, 8.0, 0.0]),
        B=1e-5 * _unit(rng.normal(size=3)),
    )
    return RWCase(
        axis_raw=axis_raw,
        max_torque=10.0,
        J=0.22,
        h=-3.1,
        h_max=3.8,
        u=float(rng.normal()),
        x=x,
        orbital_state=orbital_state,
    )


def _make_rw(case: RWCase, *, bias: Bias | None = None, noise: Noise | None = None) -> RW:
    return RW(
        axis=case.axis_raw,
        max_torque=case.max_torque,
        J=case.J,
        h=case.h,
        h_max=case.h_max,
        bias=bias,
        noise=noise,
    )


def _torque_row(case: RWCase) -> np.ndarray:
    return case.axis.reshape(1, 3)


def _zero_torque_hessian(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape)


def _zero_storage_hessian(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape)


def _squeeze_last(arr: np.ndarray) -> np.ndarray:
    return arr[..., 0]


def _expected_rw_sequence(
    case: RWCase,
    *,
    times: list[float],
    bias0: float = 0.0,
    std_bias: float = 0.0,
    noise0: float = 0.0,
    std_noise: float = 0.0,
    storage: bool = False,
) -> list[np.ndarray | float]:
    outputs = []
    bias_curr = float(bias0)
    noise_curr = float(noise0)
    bias_active = not (bias0 == 0.0 and std_bias == 0.0)
    noise_active = not (noise0 == 0.0 and std_noise == 0.0)
    last_bias_time = None

    for time in times:
        if bias_active:
            if last_bias_time is None:
                last_bias_time = time
            else:
                dt_sec = (time - last_bias_time) * TimeConstants.cent2sec
                if dt_sec > 0:
                    bias_curr = np.random.normal(loc=bias_curr, scale=std_bias * np.sqrt(dt_sec))
                    last_bias_time = time

        if noise_active:
            noise_curr = np.random.normal(loc=0.0, scale=std_noise)

        effective = case.u
        if bias_active:
            effective += bias_curr
        if noise_active:
            effective += noise_curr

        outputs.append(-effective if storage else case.axis * effective)

    return outputs


def _projected_torque_hessian_clean(case: RWCase, direction: np.ndarray) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        x = np.asarray(values[1:8])
        h = values[8]
        rw = RW(axis=case.axis_raw, max_torque=case.max_torque, J=case.J, h=h, h_max=case.h_max)
        return float(np.dot(rw.torque(u=u, x=x, os=case.orbital_state), direction))

    point = np.concatenate([[case.u], case.x, [case.h]])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def _projected_torque_hessian_biased(case: RWCase, direction: np.ndarray, bias_value: float) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        b = values[1]
        x = np.asarray(values[2:9])
        h = values[9]
        rw = RW(
            axis=case.axis_raw,
            max_torque=case.max_torque,
            J=case.J,
            h=h,
            h_max=case.h_max,
            bias=Bias(bias=b, std_bias=0.0),
        )
        return float(np.dot(rw.torque(u=u, x=x, os=case.orbital_state), direction))

    point = np.concatenate([[case.u], [bias_value], case.x, [case.h]])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def _storage_hessian_clean(case: RWCase) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        x = np.asarray(values[1:8])
        h = values[8]
        rw = RW(axis=case.axis_raw, max_torque=case.max_torque, J=case.J, h=h, h_max=case.h_max)
        return float(rw.storage_torque(u=u, x=x, os=case.orbital_state))

    point = np.concatenate([[case.u], case.x, [case.h]])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def _storage_hessian_biased(case: RWCase, bias_value: float) -> np.ndarray:
    def scalarized(values: np.ndarray) -> float:
        u = values[0]
        b = values[1]
        x = np.asarray(values[2:9])
        h = values[9]
        rw = RW(
            axis=case.axis_raw,
            max_torque=case.max_torque,
            J=case.J,
            h=h,
            h_max=case.h_max,
            bias=Bias(bias=b, std_bias=0.0),
        )
        return float(rw.storage_torque(u=u, x=x, os=case.orbital_state))

    point = np.concatenate([[case.u], [bias_value], case.x, [case.h]])
    return np.array(nd.Hessian(scalarized)(point.tolist()))


def test_rw_setup_normalizes_axis_and_copies_error_models() -> None:
    case = _make_case(0)
    bias = Bias(bias=0.13, std_bias=0.03)
    noise = Noise(noise=0.0, std_noise=0.24)

    rw = _make_rw(case, bias=bias, noise=noise)

    np.testing.assert_allclose(rw.axis, case.axis)
    assert rw.u_max == case.max_torque
    assert rw.J == case.J
    assert rw.h == case.h
    assert rw.h_max == case.h_max
    assert rw.bias is not bias
    assert rw.noise is not noise
    np.testing.assert_allclose(rw.bias.bias, bias.bias)
    np.testing.assert_allclose(rw.bias.std_bias, bias.std_bias)
    np.testing.assert_allclose(rw.noise.noise, noise.noise)
    np.testing.assert_allclose(rw.noise.std_noise, noise.std_noise)


def test_rw_clean_torque_matches_closed_form() -> None:
    case = _make_case(1)
    rw = _make_rw(case)

    np.testing.assert_allclose(rw.torque(u=case.u, x=case.x, os=case.orbital_state), case.axis * case.u)


def test_rw_biased_torque_matches_closed_form() -> None:
    case = _make_case(2)
    bias = Bias(bias=0.17, std_bias=0.0)
    rw = _make_rw(case, bias=bias)

    np.testing.assert_allclose(
        rw.torque(u=case.u, x=case.x, os=case.orbital_state),
        case.axis * (case.u + bias.get_bias(case.orbital_state.J2000)),
    )


def test_rw_clean_storage_torque_matches_closed_form() -> None:
    case = _make_case(3)
    rw = _make_rw(case)

    assert rw.storage_torque(u=case.u, x=case.x, os=case.orbital_state) == -case.u


def test_rw_biased_storage_torque_matches_closed_form() -> None:
    case = _make_case(4)
    bias = Bias(bias=0.21, std_bias=0.0)
    rw = _make_rw(case, bias=bias)

    assert rw.storage_torque(u=case.u, x=case.x, os=case.orbital_state) == -(case.u + bias.get_bias(case.orbital_state.J2000))


def test_rw_clean_torque_jacobians_match_exact_and_numeric() -> None:
    case = _make_case(5)
    rw = _make_rw(case)

    numeric_du = np.array(nd.Jacobian(lambda value: rw.torque(u=value, x=case.x, os=case.orbital_state))(case.u)).T
    numeric_dx = np.array(
        nd.Jacobian(lambda value: rw.torque(u=case.u, x=np.asarray(value), os=case.orbital_state))(case.x.tolist())
    ).T
    numeric_dh = np.array(
        nd.Jacobian(
            lambda value: RW(axis=case.axis_raw, max_torque=case.max_torque, J=case.J, h=value, h_max=case.h_max).torque(
                u=case.u, x=case.x, os=case.orbital_state
            )
        )(case.h)
    ).T

    np.testing.assert_allclose(rw.dtorq__du(u=case.u, x=case.x, os=case.orbital_state), _torque_row(case))
    np.testing.assert_allclose(rw.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 3)))
    np.testing.assert_allclose(rw.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 3)))
    np.testing.assert_allclose(rw.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 3)))

    np.testing.assert_allclose(numeric_du, rw.dtorq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, rw.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dh, rw.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state))


def test_rw_biased_torque_jacobians_match_exact_and_numeric() -> None:
    case = _make_case(6)
    bias = Bias(bias=0.15, std_bias=0.0)
    rw = _make_rw(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric_du = np.array(nd.Jacobian(lambda value: rw.torque(u=value, x=case.x, os=case.orbital_state))(case.u)).T
    numeric_db = np.array(
        nd.Jacobian(
            lambda value: RW(
                axis=case.axis_raw,
                max_torque=case.max_torque,
                J=case.J,
                h=case.h,
                h_max=case.h_max,
                bias=Bias(bias=value, std_bias=0.0),
            ).torque(u=case.u, x=case.x, os=case.orbital_state)
        )(bias_value)
    ).T
    numeric_dx = np.array(
        nd.Jacobian(lambda value: rw.torque(u=case.u, x=np.asarray(value), os=case.orbital_state))(case.x.tolist())
    ).T
    numeric_dh = np.array(
        nd.Jacobian(
            lambda value: RW(
                axis=case.axis_raw,
                max_torque=case.max_torque,
                J=case.J,
                h=value,
                h_max=case.h_max,
                bias=Bias(bias=bias_value, std_bias=0.0),
            ).torque(u=case.u, x=case.x, os=case.orbital_state)
        )(case.h)
    ).T

    np.testing.assert_allclose(rw.dtorq__du(u=case.u, x=case.x, os=case.orbital_state), _torque_row(case))
    np.testing.assert_allclose(rw.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state), _torque_row(case))
    np.testing.assert_allclose(rw.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 3)))
    np.testing.assert_allclose(rw.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 3)))

    np.testing.assert_allclose(numeric_du, rw.dtorq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_db, rw.dtorq__dbias(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, rw.dtorq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dh, rw.dtorq__dh(u=case.u, x=case.x, os=case.orbital_state))


def test_rw_clean_torque_hessian_blocks_are_zero() -> None:
    case = _make_case(7)
    rw = _make_rw(case)

    np.testing.assert_allclose(rw.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 0, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((0, 0, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((0, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((0, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((7, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((7, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))


def test_rw_biased_torque_hessian_blocks_are_zero() -> None:
    case = _make_case(8)
    rw = _make_rw(case, bias=Bias(bias=0.18, std_bias=0.0))

    np.testing.assert_allclose(rw.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((7, 7, 3)))
    np.testing.assert_allclose(rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((7, 1, 3)))
    np.testing.assert_allclose(rw.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state), _zero_torque_hessian((1, 1, 3)))


@pytest.mark.parametrize("direction", MathConstants.unitvecs)
def test_rw_clean_projected_torque_hessian_matches_finite_difference(direction: np.ndarray) -> None:
    case = _make_case(9)
    rw = _make_rw(case)

    numeric = _projected_torque_hessian_clean(case, direction)
    expected = np.block(
        [
            [
                rw.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                rw.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                (rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                rw.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-14)


@pytest.mark.parametrize("direction", MathConstants.unitvecs)
def test_rw_biased_projected_torque_hessian_matches_finite_difference(direction: np.ndarray) -> None:
    case = _make_case(10)
    bias = Bias(bias=0.12, std_bias=0.0)
    rw = _make_rw(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric = _projected_torque_hessian_biased(case, direction, bias_value)
    expected = np.block(
        [
            [
                rw.ddtorq__dudu(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (rw.ddtorq__dudbias(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                rw.ddtorq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (rw.ddtorq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                (rw.ddtorq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                rw.ddtorq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state) @ direction,
                rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
            [
                (rw.ddtorq__dudh(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                (rw.ddtorq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                (rw.ddtorq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state) @ direction).T,
                rw.ddtorq__dhdh(u=case.u, x=case.x, os=case.orbital_state) @ direction,
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-14)


def test_rw_clean_storage_jacobians_match_exact_and_numeric() -> None:
    case = _make_case(11)
    rw = _make_rw(case)

    numeric_du = np.array(nd.Jacobian(lambda value: rw.storage_torque(u=value, x=case.x, os=case.orbital_state))(case.u))
    numeric_dx = np.array(
        nd.Jacobian(lambda value: rw.storage_torque(u=case.u, x=np.asarray(value), os=case.orbital_state))(case.x.tolist())
    ).T
    numeric_dh = np.array(
        nd.Jacobian(
            lambda value: RW(axis=case.axis_raw, max_torque=case.max_torque, J=case.J, h=value, h_max=case.h_max).storage_torque(
                u=case.u, x=case.x, os=case.orbital_state
            )
        )(case.h)
    )

    np.testing.assert_allclose(rw.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state), -np.ones((1, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dbias(u=case.u, x=case.x, os=case.orbital_state), np.zeros((0, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1)))

    np.testing.assert_allclose(numeric_du, rw.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, rw.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dh, rw.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state))


def test_rw_biased_storage_jacobians_match_exact_and_numeric() -> None:
    case = _make_case(12)
    bias = Bias(bias=0.14, std_bias=0.0)
    rw = _make_rw(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric_du = np.array(nd.Jacobian(lambda value: rw.storage_torque(u=value, x=case.x, os=case.orbital_state))(case.u))
    numeric_db = np.array(
        nd.Jacobian(
            lambda value: RW(
                axis=case.axis_raw,
                max_torque=case.max_torque,
                J=case.J,
                h=case.h,
                h_max=case.h_max,
                bias=Bias(bias=value, std_bias=0.0),
            ).storage_torque(u=case.u, x=case.x, os=case.orbital_state)
        )(bias_value)
    )
    numeric_dx = np.array(
        nd.Jacobian(lambda value: rw.storage_torque(u=case.u, x=np.asarray(value), os=case.orbital_state))(case.x.tolist())
    ).T
    numeric_dh = np.array(
        nd.Jacobian(
            lambda value: RW(
                axis=case.axis_raw,
                max_torque=case.max_torque,
                J=case.J,
                h=value,
                h_max=case.h_max,
                bias=Bias(bias=bias_value, std_bias=0.0),
            ).storage_torque(u=case.u, x=case.x, os=case.orbital_state)
        )(case.h)
    )

    np.testing.assert_allclose(rw.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state), -np.ones((1, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dbias(u=case.u, x=case.x, os=case.orbital_state), -np.ones((1, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state), np.zeros((7, 1)))
    np.testing.assert_allclose(rw.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state), np.zeros((1, 1)))

    np.testing.assert_allclose(numeric_du, rw.dstor_torq__du(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_db, rw.dstor_torq__dbias(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dx, rw.dstor_torq__dbasestate(u=case.u, x=case.x, os=case.orbital_state))
    np.testing.assert_allclose(numeric_dh, rw.dstor_torq__dh(u=case.u, x=case.x, os=case.orbital_state))


def test_rw_clean_storage_hessian_blocks_are_zero() -> None:
    case = _make_case(13)
    rw = _make_rw(case)

    np.testing.assert_allclose(rw.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 0, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((0, 0, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((0, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((0, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((7, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((7, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))


def test_rw_biased_storage_hessian_blocks_are_zero() -> None:
    case = _make_case(14)
    rw = _make_rw(case, bias=Bias(bias=0.19, std_bias=0.0))

    np.testing.assert_allclose(rw.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((7, 7, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((7, 1, 1)))
    np.testing.assert_allclose(rw.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state), _zero_storage_hessian((1, 1, 1)))


def test_rw_clean_storage_hessian_matches_finite_difference() -> None:
    case = _make_case(15)
    rw = _make_rw(case)

    numeric = _storage_hessian_clean(case)
    expected = np.block(
        [
            [
                _squeeze_last(rw.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
            [
                _squeeze_last(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
            [
                _squeeze_last(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-14)


def test_rw_biased_storage_hessian_matches_finite_difference() -> None:
    case = _make_case(16)
    bias = Bias(bias=0.16, std_bias=0.0)
    rw = _make_rw(case, bias=bias)
    bias_value = bias.get_bias(case.orbital_state.J2000)

    numeric = _storage_hessian_biased(case, bias_value)
    expected = np.block(
        [
            [
                _squeeze_last(rw.ddstor_torq__dudu(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
            [
                _squeeze_last(rw.ddstor_torq__dudbias(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbiasdbias(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
            [
                _squeeze_last(rw.ddstor_torq__dudbasestate(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbiasdbasestate(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbasestatedbasestate(u=case.u, x=case.x, os=case.orbital_state)),
                _squeeze_last(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
            [
                _squeeze_last(rw.ddstor_torq__dudh(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbiasdh(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dbasestatedh(u=case.u, x=case.x, os=case.orbital_state)).T,
                _squeeze_last(rw.ddstor_torq__dhdh(u=case.u, x=case.x, os=case.orbital_state)),
            ],
        ]
    )

    np.testing.assert_allclose(numeric, expected, atol=1e-14)


def test_rw_bias_does_not_change_without_time_advance() -> None:
    case = _make_case(17)
    rw = _make_rw(case, bias=Bias(bias=0.09, std_bias=0.03))

    first = rw.torque(u=case.u, x=case.x, os=case.orbital_state)
    repeated = [rw.torque(u=case.u, x=case.x, os=case.orbital_state) for _ in range(3)]

    for value in repeated:
        np.testing.assert_allclose(value, first)


def test_rw_bias_updates_follow_seeded_torque_sequence() -> None:
    case = _make_case(18)
    bias0 = 0.08
    std_bias = 0.03
    rw = _make_rw(case, bias=Bias(bias=bias0, std_bias=std_bias))
    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(123)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(rw.torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(123)
    expected = _expected_rw_sequence(case, times=times, bias0=bias0, std_bias=std_bias)
    np.testing.assert_allclose(actual, expected)


def test_rw_noise_updates_follow_seeded_torque_sequence() -> None:
    case = _make_case(19)
    std_noise = 0.24
    rw = _make_rw(case, noise=Noise(noise=0.0, std_noise=std_noise))
    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(456)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(rw.torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(456)
    expected = _expected_rw_sequence(case, times=times, noise0=0.0, std_noise=std_noise)
    np.testing.assert_allclose(actual, expected)


def test_rw_bias_and_noise_updates_follow_seeded_torque_sequence() -> None:
    case = _make_case(20)
    bias0 = 0.05
    std_bias = 0.03
    std_noise = 0.18
    rw = _make_rw(
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
        actual.append(rw.torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(789)
    expected = _expected_rw_sequence(
        case,
        times=times,
        bias0=bias0,
        std_bias=std_bias,
        noise0=0.0,
        std_noise=std_noise,
    )
    np.testing.assert_allclose(actual, expected)


def test_rw_bias_updates_follow_seeded_storage_sequence() -> None:
    case = _make_case(21)
    bias0 = 0.07
    std_bias = 0.03
    rw = _make_rw(case, bias=Bias(bias=bias0, std_bias=std_bias))
    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(321)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(rw.storage_torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(321)
    expected = _expected_rw_sequence(case, times=times, bias0=bias0, std_bias=std_bias, storage=True)
    np.testing.assert_allclose(actual, expected)


def test_rw_noise_updates_follow_seeded_storage_sequence() -> None:
    case = _make_case(22)
    std_noise = 0.16
    rw = _make_rw(case, noise=Noise(noise=0.0, std_noise=std_noise))
    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(654)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(rw.storage_torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(654)
    expected = _expected_rw_sequence(case, times=times, noise0=0.0, std_noise=std_noise, storage=True)
    np.testing.assert_allclose(actual, expected)


def test_rw_bias_and_noise_updates_follow_seeded_storage_sequence() -> None:
    case = _make_case(23)
    bias0 = 0.04
    std_bias = 0.03
    std_noise = 0.11
    rw = _make_rw(
        case,
        bias=Bias(bias=bias0, std_bias=std_bias),
        noise=Noise(noise=0.0, std_noise=std_noise),
    )
    dt = 0.5 * TimeConstants.sec2cent
    times = [case.orbital_state.J2000 + idx * dt for idx in range(4)]

    np.random.seed(987)
    actual = []
    for time in times:
        case.orbital_state.J2000 = time
        actual.append(rw.storage_torque(u=case.u, x=case.x, os=case.orbital_state))

    np.random.seed(987)
    expected = _expected_rw_sequence(
        case,
        times=times,
        bias0=bias0,
        std_bias=std_bias,
        noise0=0.0,
        std_noise=std_noise,
        storage=True,
    )
    np.testing.assert_allclose(actual, expected)
