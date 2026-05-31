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

    assert np.all(np.isclose( rw.ddtorq__dudu(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(u=t0, x=x0, os=os).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(u=t0, x=x0, os=os) ,np.zeros((1,0,3)) ))
    assert np.all(rw.ddtorq__dudbias(u=t0, x=x0, os=os).shape==(1,0,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(u=t0, x=x0, os=os) ,np.zeros((1,7,3))))
    assert np.all(rw.ddtorq__dudbasestate(u=t0, x=x0, os=os).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(u=t0, x=x0, os=os).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(u=t0, x=x0, os=os) ,np.zeros((0,0,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(u=t0, x=x0, os=os).shape==(0,0,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os) ,np.zeros((0,7,3)) ))#np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os).shape==(0,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(u=t0, x=x0, os=os) ,np.zeros((0,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(u=t0, x=x0, os=os).shape==(0,1,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(u=t0, x=x0, os=os).shape==(1,1,3))

    assert np.all(rw.storage_torque(u=t0, x=x0, os=os)  == -t0)
    assert np.all(np.isclose( rw.dstor_torq__du(u=t0, x=x0, os=os) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbias(u=t0, x=x0, os=os) , np.zeros((0,1)) ))
    assert np.all(rw.dstor_torq__dbias(u=t0, x=x0, os=os).shape == (0,1))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(u=t0, x=x0, os=os) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(u=t0, x=x0, os=os).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(u=t0, x=x0, os=os) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(u=t0, x=x0, os=os).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(u=t0, x=x0, os=os).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(u=t0, x=x0, os=os) ,np.zeros((1,0,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(u=t0, x=x0, os=os).shape==(1,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(u=t0, x=x0, os=os).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(u=t0, x=x0, os=os) ,np.zeros((0,0 ,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(u=t0, x=x0, os=os).shape==(0,0,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(u=t0, x=x0, os=os) , np.zeros((0,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(u=t0, x=x0, os=os).shape==(0,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(u=t0, x=x0, os=os) ,np.zeros((0,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(u=t0, x=x0, os=os).shape==(0,1,1))

    dxdx = np.zeros((7,7,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(u=t0, x=x0, os=os).shape==(1,1,1))

    ufun = lambda c: rw.storage_torque(u=c, x=x0, os=os).item()
    xfun = lambda c: rw.storage_torque(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()
    hfun = lambda c: rw.storage_torque(u=t0, x=x0, os=os).item()
    bfun = lambda c: rw.storage_torque(u=t0, x=x0, os=os).item()

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist()))
    Jufun = np.array(nd.Jacobian(ufun)(t0))
    Jbfun = np.array(nd.Jacobian(bfun)(20000))
    Jhfun = np.array(nd.Jacobian(hfun)(h0))

    assert np.allclose(Jxfun, rw.dstor_torq__dbasestate(u=t0, x=x0, os=os))
    assert np.allclose(Jufun, rw.dstor_torq__du(u=t0, x=x0, os=os))
    assert np.allclose(Jbfun, rw.dstor_torq__dbias(u=t0, x=x0, os=os))
    assert np.allclose(Jhfun, rw.dstor_torq__dh(u=t0, x=x0, os=os))

    for j in MathConstants.unitvecs:
        ufunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        # ufunjjb = lambda c: rw.dstor_torq__dbias(u=t0, x=x0, os=os)
        ufunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os)
        ufunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        xfunjju = lambda c: rw.dstor_torq__du(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()
        xfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
        xfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()

        hfunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        hfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os)
        hfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        bfunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        bfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os).item()
        bfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jxfunjjx , rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jxfunjjh , rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , rw.ddstor_torq__dudu(u=t0, x=x0, os=os))
        assert np.allclose( Jufunjjx , rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jufunjjh , rw.ddstor_torq__dudh(u=t0, x=x0, os=os))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(h0))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(h0))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(h0))
        assert np.allclose( Jhfunjju , rw.ddstor_torq__dudh(u=t0, x=x0, os=os))
        assert np.allclose( Jhfunjjx , rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os))
        assert np.allclose( Jhfunjjh , rw.ddstor_torq__dhdh(u=t0, x=x0, os=os))


def test_RW_torque_etc_bias():
    # --- Setup ---
    ax = random_n_unit_vec(3) * 3
    ax = ax.copy()
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    biast = random_n_unit_vec(3)[1]*0.1
    biasv = biast.copy()
    std_bias = 0.03
    bias = Bias(bias=biast, std_bias=std_bias)

    # No noise
    rw = RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max, bias=bias, noise=None)
    rws = [rw]

    t0 = random_n_unit_vec(3)[0]  # scalar command
    B_ECI = 1e-5 * random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    sat = Satellite(actuators=rws)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    # Body-frame vectors (not used by RW torque, but kept for parity with other tests)
    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B @ R
    B_B = rmat_ECI2B @ B
    S_B = rmat_ECI2B @ S
    V_B = rmat_ECI2B @ V
    dR_B__dq = drotmatTvecdq(q0, R)
    dB_B__dq = drotmatTvecdq(q0, B)
    dV_B__dq = drotmatTvecdq(q0, V)
    dS_B__dq = drotmatTvecdq(q0, S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0, R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0, B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0, V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0, S)

    assert np.all(np.isclose(ax/3*(t0+biast),rw.torque(u=t0, x=x0, os=os)))

    # --- Finite-difference helpers (new interface) ---
    # vary u
    ufun = lambda c: rw.torque(u=c, x=x0, os=os)
    xfun = lambda c: rw.torque(u=t0, x=np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6]]), os=os)
    hfun = lambda c: RW(axis=ax, max_torque=max_torque, J=J, h=c, h_max=h_max, bias=None, noise=None).torque(u=t0, x=x0, os=os)
    bfun = lambda c: RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max,
                        bias=Bias(bias=c, std_bias=0.0), noise=None).torque(u=t0, x=x0, os=os)

    # --- Jacobians: numerical vs analytic ---
    Jx_num = np.array(nd.Jacobian(xfun)(x0.flatten().tolist())).T
    Ju_num = np.array(nd.Jacobian(ufun)(t0)).T
    Jh_num = np.array(nd.Jacobian(hfun)(h0)).T
    Jb_num = np.array(nd.Jacobian(bfun)(500.2)).T  # any scalar; torque shouldn't change with "bias" in this object

    Jx_ana = rw.dtorq__dbasestate(u=t0, x=x0, os=os)
    Ju_ana = rw.dtorq__du(u=t0, x=x0, os=os)
    Jb_ana = rw.dtorq__dbias(u=t0, x=x0, os=os)
    Jh_ana = rw.dtorq__dh(u=t0, x=x0, os=os)

    assert Jx_ana.shape == (7, 3)
    assert np.allclose(Jx_num, Jx_ana)
    assert np.allclose(Ju_num, Ju_ana)
    assert np.allclose(Jb_num, Jb_ana)
    assert np.allclose(Jh_num, Jh_ana)

    # dτ/dh: many RW torque models do not depend on h → zero. Match analytic convention & allow either (0,3) or (1,3)
    if Jh_ana.size == 0:
        assert Jh_ana.shape == (0, 3)
    else:
        assert Jh_ana.shape == (1, 3)
        assert np.allclose(Jh_ana, 0.0)
        assert np.allclose(Jh_num, 0.0, atol=1e-10)

    # --- Second-order derivatives (should be zero for a clean linear RW) ---
    # Scalarize via projection like in your template
    for j in MathConstants.unitvecs:
        fun_hj = lambda c: np.dot(RW(axis=ax, max_torque=max_torque, J=J, h=c[9], h_max=h_max, bias=Bias(bias=c[1], std_bias=std_bias)).torque(c[0], x=np.array([c[2],c[3],c[4],c[5],c[6],c[7],c[8]]), os=os), j).item()

        ufunjju = lambda c: np.dot(rw.dtorq__du(u=c, x=x0, os=os), j).item()
        ufunjjb = lambda c: np.dot(rw.dtorq__dbias(u=c, x=x0, os=os), j)  # empty/zero
        ufunjjx = lambda c: np.dot(rw.dtorq__dbasestate(u=c, x=x0, os=os), j)
        ufunjjh = lambda c: np.dot(rw.dtorq__dh(u=c, x=x0, os=os), j).item()

        xfunjju = lambda c: np.dot(rw.dtorq__du(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os), j).item()
        xfunjjb = lambda c: np.dot(rw.dtorq__dbias(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os), j).item()
        xfunjjx = lambda c: np.dot(rw.dtorq__dbasestate(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os), j)
        xfunjjh = lambda c: np.dot(rw.dtorq__dh(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os), j).item()

        hfunjju = lambda c: np.dot(rw.dtorq__du(u=t0, x=x0, os=os),j).item()
        hfunjjb = lambda c: np.dot(rw.dtorq__dbias(u=t0, x=x0, os=os),j)
        hfunjjx = lambda c: np.dot(rw.dtorq__dbasestate(u=t0, x=x0, os=os),j)
        hfunjjh = lambda c: np.dot(rw.dtorq__dh(u=t0, x=x0, os=os),j)

        bfunjju = lambda c: np.dot(RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max).dtorq__du(u=t0, x=x0, os=os), j).item()
        bfunjjb = lambda c: np.dot(RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max).dtorq__dbias(u=t0, x=x0, os=os), j).item()
        bfunjjx = lambda c: np.dot(RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max).dtorq__dbasestate(u=t0, x=x0, os=os), j)
        bfunjjh = lambda c: np.dot(RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max).dtorq__dh(u=t0, x=x0, os=os), j)

        # Mixed (u,x)
        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose(Jxfunjju, np.dot(rw.ddtorq__dudbasestate(u=t0, x=x0, os=os), j))
        assert np.allclose(Jxfunjjx, np.dot(rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os), j))
        assert np.allclose( Jxfunjjh , np.dot(rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os), j))

        # (u,u) and (u,x) again
        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose(Jufunjju, np.dot(rw.ddtorq__dudu(u=t0, x=x0, os=os), j))
        assert np.allclose(Jufunjjx.T, np.dot(rw.ddtorq__dudbasestate(u=t0, x=x0, os=os), j))
        assert np.allclose(Jufunjjh , np.dot(rw.ddtorq__dudh(u=t0, x=x0, os=os), j))

        # (h,·) mixed (should be zero)
        Jbfunjju = np.array(nd.Jacobian(bfunjju)(20000))
        Jbfunjjx = np.array(nd.Jacobian(bfunjjx)(20000))
        Jbfunjjh = np.array(nd.Jacobian(bfunjjh)(20000))
        assert np.allclose( Jbfunjju , np.dot( rw.ddtorq__dudbias(u=t0, x=x0, os=os),j))
        assert np.allclose( Jbfunjjx.T , np.dot( rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os),j))
        assert np.allclose( Jbfunjjh , np.dot( rw.ddtorq__dbiasdh(u=t0, x=x0, os=os),j))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(h0))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(h0))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(h0))
        assert np.allclose( Jhfunjju , np.dot( rw.ddtorq__dudh(u=t0, x=x0, os=os),j))
        assert np.allclose( Jhfunjjx , np.dot( rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os),j))
        assert np.allclose( Jhfunjjh , np.dot( rw.ddtorq__dhdh(u=t0, x=x0, os=os),j))

        Hfun = np.array(nd.Hessian(fun_hj)(np.concatenate([[t0],[biast],x0,[h0]]).flatten().tolist()))
        Hguess = np.block([[rw.ddtorq__dudu(u=t0, x=x0, os=os)@j,rw.ddtorq__dudbias(u=t0, x=x0, os=os)@j,rw.ddtorq__dudbasestate(u=t0, x=x0, os=os)@j,rw.ddtorq__dudh(u=t0, x=x0, os=os)@j],\
                            [(rw.ddtorq__dudbias(u=t0, x=x0, os=os)@j).T,rw.ddtorq__dbiasdbias(u=t0, x=x0, os=os)@j,rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os)@j,rw.ddtorq__dbiasdh(u=t0, x=x0, os=os)@j],\
                            [(rw.ddtorq__dudbasestate(u=t0, x=x0, os=os)@j).T,(rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os)@j).T,rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os)@j,rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os)@j],\
                            [(rw.ddtorq__dudh(u=t0, x=x0, os=os)@j).T,(rw.ddtorq__dbiasdh(u=t0, x=x0, os=os)@j).T,(rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os)@j).T,rw.ddtorq__dhdh(u=t0, x=x0, os=os)@j]    ])

        assert np.allclose(Hfun,Hguess)

    assert np.all(np.isclose( rw.dtorq__du(u=t0, x=x0, os=os) ,ax/3 ))
    assert np.all(np.isclose( rw.dtorq__dbias(u=t0, x=x0, os=os) ,np.zeros((0,3))))
    assert np.all(np.isclose( rw.dtorq__dbasestate(u=t0, x=x0, os=os) , np.zeros((7,3))))
    assert np.all(np.isclose( rw.dtorq__dh(u=t0, x=x0, os=os) , np.zeros((1,3))))

    assert np.all(np.isclose( rw.ddtorq__dudu(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudu(u=t0, x=x0, os=os).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbias(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudbias(u=t0, x=x0, os=os).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dudbasestate(u=t0, x=x0, os=os) ,np.zeros((1,7,3))))
    assert np.all(rw.ddtorq__dudbasestate(u=t0, x=x0, os=os).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dudh(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dudh(u=t0, x=x0, os=os).shape==(1,1,3))

    assert np.all(np.isclose( rw.ddtorq__dbiasdbias(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdbias(u=t0, x=x0, os=os).shape==(1,1,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os) ,np.zeros((1,7,3)) ))#np.expand_dims(np.vstack([np.zeros((3,3)),np.cross(ax/3,drotmatTvecdq(q0,B_ECI))]),0) ))
    assert np.all(rw.ddtorq__dbiasdbasestate(u=t0, x=x0, os=os).shape==(1,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbiasdh(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dbiasdh(u=t0, x=x0, os=os).shape==(1,1,3))

    dxdx = np.zeros((7,7,3))

    assert np.all(np.isclose( rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os) , dxdx))
    assert np.all(rw.ddtorq__dbasestatedbasestate(u=t0, x=x0, os=os).shape==(7,7,3))
    assert np.all(np.isclose( rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os) ,np.zeros((7,1,3)) ))
    assert np.all(rw.ddtorq__dbasestatedh(u=t0, x=x0, os=os).shape==(7,1,3))
    assert np.all(np.isclose( rw.ddtorq__dhdh(u=t0, x=x0, os=os) ,np.zeros((1,1,3)) ))
    assert np.all(rw.ddtorq__dhdh(u=t0, x=x0, os=os).shape==(1,1,3))

    assert np.all(rw.storage_torque(u=t0, x=x0, os=os)  == -(t0 + biast))
    assert np.all(np.isclose( rw.dstor_torq__du(u=t0, x=x0, os=os) , -1 ))
    assert np.all(np.isclose( rw.dstor_torq__dbias(u=t0, x=x0, os=os) , -1 ))
    assert np.all(rw.dstor_torq__dbias(u=t0, x=x0, os=os).shape == (1,1))
    assert np.all(np.isclose( rw.dstor_torq__dbasestate(u=t0, x=x0, os=os) , np.zeros((7,1))))
    assert np.all(rw.dstor_torq__dbasestate(u=t0, x=x0, os=os).shape == (7,1))
    assert np.all(np.isclose( rw.dstor_torq__dh(u=t0, x=x0, os=os) , np.zeros((1,1))))
    assert np.all(rw.dstor_torq__dh(u=t0, x=x0, os=os).shape==(1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dudu(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudu(u=t0, x=x0, os=os).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbias(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudbias(u=t0, x=x0, os=os).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os) ,np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dudh(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dudh(u=t0, x=x0, os=os).shape==(1,1,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbias(u=t0, x=x0, os=os) ,np.zeros((1, 1, 1)) ))
    assert np.all(rw.ddstor_torq__dbiasdbias(u=t0, x=x0, os=os).shape==(1,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdbasestate(u=t0, x=x0, os=os) , np.zeros((1,7,1))))
    assert np.all(rw.ddstor_torq__dbiasdbasestate(u=t0, x=x0, os=os).shape==(1,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbiasdh(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dbiasdh(u=t0, x=x0, os=os).shape==(1,1,1))

    dxdx = np.zeros((7,7,1))

    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os) , dxdx))
    assert np.all(rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os).shape==(7,7,1))
    assert np.all(np.isclose( rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os) ,np.zeros((7,1,1)) ))
    assert np.all(rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os).shape==(7,1,1))
    assert np.all(np.isclose( rw.ddstor_torq__dhdh(u=t0, x=x0, os=os) ,np.zeros((1,1,1)) ))
    assert np.all(rw.ddstor_torq__dhdh(u=t0, x=x0, os=os).shape==(1,1,1))

    ufun = lambda c: rw.storage_torque(u=c, x=x0, os=os).item()
    xfun = lambda c: rw.storage_torque(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()
    hfun = lambda c: RW(axis=ax, max_torque=max_torque, J=J, h=c, h_max=h_max, bias=None, noise=None).storage_torque(u=t0, x=x0, os=os)
    bfun = lambda c: RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max,
                        bias=Bias(bias=c, std_bias=std_bias), noise=None).storage_torque(u=t0, x=x0, os=os)

    Jxfun = np.array(nd.Jacobian(xfun)(x0.flatten().tolist()))
    Jufun = np.array(nd.Jacobian(ufun)(t0))
    Jbfun = np.array(nd.Jacobian(bfun)(biast))
    Jhfun = np.array(nd.Jacobian(hfun)(h0))

    assert np.allclose(Jxfun, rw.dstor_torq__dbasestate(u=t0, x=x0, os=os))
    assert np.allclose(Jufun, rw.dstor_torq__du(u=t0, x=x0, os=os))
    assert np.allclose(Jbfun, rw.dstor_torq__dbias(u=t0, x=x0, os=os))
    assert np.allclose(Jhfun, rw.dstor_torq__dh(u=t0, x=x0, os=os))

    for j in MathConstants.unitvecs:
        ufunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        # ufunjjb = lambda c: rw.dstor_torq__dbias(u=t0, x=x0, os=os)
        ufunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os)
        ufunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        xfunjju = lambda c: rw.dstor_torq__du(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()
        xfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os)
        xfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=np.array([c[0],c[1],c[2],c[3],c[4],c[5],c[6]]), os=os).item()

        hfunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        hfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os)
        hfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        bfunjju = lambda c: rw.dstor_torq__du(u=t0, x=x0, os=os).item()
        bfunjjx = lambda c: rw.dstor_torq__dbasestate(u=t0, x=x0, os=os).item()
        bfunjjh = lambda c: rw.dstor_torq__dh(u=t0, x=x0, os=os).item()

        Jxfunjju = np.array(nd.Jacobian(xfunjju)(x0.flatten().tolist()))
        Jxfunjjx = np.array(nd.Jacobian(xfunjjx)(x0.flatten().tolist()))
        Jxfunjjh = np.array(nd.Jacobian(xfunjjh)(x0.flatten().tolist()))
        assert np.allclose( Jxfunjju , rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jxfunjjx , rw.ddstor_torq__dbasestatedbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jxfunjjh , rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os))

        Jufunjju = np.array(nd.Jacobian(ufunjju)(t0))
        Jufunjjx = np.array(nd.Jacobian(ufunjjx)(t0))
        Jufunjjh = np.array(nd.Jacobian(ufunjjh)(t0))
        assert np.allclose( Jufunjju , rw.ddstor_torq__dudu(u=t0, x=x0, os=os))
        assert np.allclose( Jufunjjx , rw.ddstor_torq__dudbasestate(u=t0, x=x0, os=os))
        assert np.allclose( Jufunjjh , rw.ddstor_torq__dudh(u=t0, x=x0, os=os))

        Jhfunjju = np.array(nd.Jacobian(hfunjju)(h0))
        Jhfunjjx = np.array(nd.Jacobian(hfunjjx)(h0))
        Jhfunjjh = np.array(nd.Jacobian(hfunjjh)(h0))
        assert np.allclose( Jhfunjju , rw.ddstor_torq__dudh(u=t0, x=x0, os=os))
        assert np.allclose( Jhfunjjx , rw.ddstor_torq__dbasestatedh(u=t0, x=x0, os=os))
        assert np.allclose( Jhfunjjh , rw.ddstor_torq__dhdh(u=t0, x=x0, os=os))


def test_RW_torque_bias_KS():
    ax = random_n_unit_vec(3)
    ax = ax.copy()
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    biast = random_n_unit_vec(3)[1]*0.1
    biasv = biast.copy()
    std_bias = 0.03
    bias = Bias(bias=biast, std_bias=std_bias)

    # No noise
    rw = RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max, bias=bias, noise=None)
    rws = [rw]

    t0 = random_n_unit_vec(3)[0]  # scalar command
    B_ECI = 1e-5 * random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    sat = Satellite(actuators=rws)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    R = os.R
    V = os.V
    B = B_ECI
    S = os.S
    rho = os.rho
    rmat_ECI2B = rot_mat(q0).T
    R_B = rmat_ECI2B@R
    B_B = rmat_ECI2B@B
    S_B = rmat_ECI2B@S
    V_B = rmat_ECI2B@V
    dR_B__dq = drotmatTvecdq(q0,R)
    dB_B__dq = drotmatTvecdq(q0,B)
    dV_B__dq = drotmatTvecdq(q0,V)
    dS_B__dq = drotmatTvecdq(q0,S)
    ddR_B__dqdq = ddrotmatTvecdqdq(q0,R)
    ddB_B__dqdq = ddrotmatTvecdqdq(q0,B)
    ddV_B__dqdq = ddrotmatTvecdqdq(q0,V)
    ddS_B__dqdq = ddrotmatTvecdqdq(q0,S)

    # No change to bias if time in Orbital_State does not progress
    N = 1000
    test_torq = rw.torque(u=t0, x=x0, os=os)
    opts = [rw.torque(u=t0, x=x0, os=os) for j in range(N)]
    assert np.all([np.allclose(test_torq, j) for j in opts])

    # Change to bias
    N = 1000
    test_torq = rw.torque(u=t0, x=x0, os=os)
    torq_drift = []
    for j in range(N):
        os.J2000 += 0.5*TimeConstants.sec2cent
        torque1 = rw.torque(u=t0, x=x0, os=os)
        os.J2000 += 0.5*TimeConstants.sec2cent
        torque2 = rw.torque(u=t0, x=x0, os=os)
        torq_drift.append(torque1 - torque2)

    exp_dist = [ax*np.random.normal(0,std_bias*np.sqrt(0.5)) for j in range(N)]
    
    ks0 = kstest([j[0] for j in torq_drift],[j[0] for j in exp_dist])
    ks1 = kstest([j[1] for j in torq_drift],[j[1] for j in exp_dist])
    ks2 = kstest([j[2] for j in torq_drift],[j[2] for j in exp_dist])

    ind = 0
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks0.pvalue>0.1 or np.abs(ks0.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 1
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks1.pvalue>0.1 or np.abs(ks1.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))
    ind = 2
    data_a = torq_drift
    data_b = exp_dist
    hist = np.histogram([dd[ind] for dd in data_a],bins='auto')
    hist_edges = hist[1]
    hist_a = np.cumsum(hist[0]).tolist()
    hist_b = [sum([dd[ind] for dd in data_b]<ee) for ee in hist_edges[1:]]
    graph_data = [hist_a,hist_b]
    print(plot(graph_data,{'height':20}))
    assert ks2.pvalue>0.1 or np.abs(ks2.statistic)<(np.sqrt((1/N)*-0.5*np.log(0.5*1e-5)))


def test_RW_storage_torque_bias_KS():
    # --- Setup ---
    ax = random_n_unit_vec(3)                # RW axis (will be normalized by RW)
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    biast = random_n_unit_vec(3)[1]*0.1      # initial bias value (scalar)
    std_bias = 0.03                          # bias RW diffusion [per sqrt(sec)]
    bias = Bias(bias=biast, std_bias=std_bias)

    # RW with bias, no output noise
    rw = RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max, bias=bias, noise=None)

    u0 = random_n_unit_vec(3)[0]             # scalar torque command
    B_ECI = 1e-5 * random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    w0 = 0.05 * random_n_unit_vec(3)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    # Sanity: no time advance ⇒ no bias change ⇒ identical storage torque
    N_same = 200
    st0 = rw.storage_torque(u=u0, x=x0, os=os)
    st_reps = [rw.storage_torque(u=u0, x=x0, os=os) for _ in range(N_same)]
    assert np.all([np.allclose(st0, s) for s in st_reps])

    # --- Bias drift KS test (advance J2000 in two independent half-steps) ---
    N = 1000
    st_drift = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        s1 = rw.storage_torque(u=u0, x=x0, os=os)   # uses b(t + Δt/2)

        os.J2000 += 0.5 * TimeConstants.sec2cent
        s2 = rw.storage_torque(u=u0, x=x0, os=os)   # uses b(t + Δt)

        # storage_torque = -(u + b), so s1 - s2 = -(b1 - b2) = (b2 - b1)
        st_drift.append(s1 - s2)

    st_drift = np.asarray(st_drift).reshape(-1)     # shape (N,)

    # Theoretical: (b2 - b1) ~ N(0, std_bias^2 * 0.5), so scalar Normal:
    exp_dist = np.random.normal(0.0, std_bias * np.sqrt(0.5), size=N)

    # Two-sample KS (recommended)
    ks = ks_2samp(st_drift, exp_dist)
    threshold = np.sqrt((1/N) * -0.5 * np.log(0.5 * 1e-5))

    # Accept if either p large or D small vs analytic bound
    assert ks.pvalue > 0.1 or abs(ks.statistic) < threshold


def test_RW_torque_noise_KS():
    ax = random_n_unit_vec(3)
    ax = ax.copy()
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    std_noise = 0.03
    noise = Noise(noise=0.0, std_noise=std_noise)

    # No bias
    rw = RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max, noise=noise)
    rws = [rw]

    t0 = random_n_unit_vec(3)[0]
    B_ECI = 1e-5 * random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    R = rot_mat(q0)
    w0 = 0.05 * random_n_unit_vec(3)
    sat = Satellite(actuators=rws)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    # collect differences over two half-steps
    N = 1000
    torq_drift = []
    for _ in range(N):
        os.J2000 += 0.5*TimeConstants.sec2cent
        t1 = rw.torque(u=t0, x=x0, os=os)      # uses noise at t+0.5
        os.J2000 += 0.5*TimeConstants.sec2cent
        t2 = rw.torque(u=t0, x=x0, os=os)      # uses noise at t+1.0
        torq_drift.append(t1 - t2)

    torq_drift = np.stack(torq_drift, axis=0)   # (N,3)

    # RW noise is scalar along the wheel axis: Δτ = (n1 - n2) * axis
    # For the same time-driven model as your MTQ test, std = std_noise (not sqrt(2))
    a = rw.axis.reshape(1, 3)                   # unit axis used internally
    exp_dist = np.random.normal(0.0, np.sqrt(2)*std_noise, size=(N, 1)) * a  # (N,3)

    ks0 = ks_2samp(torq_drift[:, 0], exp_dist[:, 0])
    ks1 = ks_2samp(torq_drift[:, 1], exp_dist[:, 1])
    ks2 = ks_2samp(torq_drift[:, 2], exp_dist[:, 2])

    threshold = np.sqrt((1/N) * -0.5 * np.log(0.5 * 1e-5))
    assert ks0.pvalue > 0.05 or abs(ks0.statistic) < threshold
    assert ks1.pvalue > 0.05 or abs(ks1.statistic) < threshold
    assert ks2.pvalue > 0.05 or abs(ks2.statistic) < threshold

def test_RW_storage_torque_noise_KS():
    ax = random_n_unit_vec(3)
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    std_noise = 0.03
    noise = Noise(noise=0.0, std_noise=std_noise)

    rw = RW(axis=ax, max_torque=max_torque, J=J, h=h0, h_max=h_max, noise=noise)

    t0 = random_n_unit_vec(3)[0]
    B_ECI = 1e-5 * random_n_unit_vec(3)
    q0 = random_n_unit_vec(4)
    w0 = 0.05 * random_n_unit_vec(3)

    x0 = np.hstack((w0, q0))
    ephem = Ephemeris()
    os = Orbital_State(ephem=ephem, J2000=0.22, R=np.array([7000, 0, 0]), V=np.array([0, 8, 0]), B=B_ECI)

    # Collect scalar differences from two half-steps
    N = 1000
    drifts = []
    for _ in range(N):
        os.J2000 += 0.5 * TimeConstants.sec2cent
        s1 = rw.storage_torque(u=t0, x=x0, os=os)   # scalar
        os.J2000 += 0.5 * TimeConstants.sec2cent
        s2 = rw.storage_torque(u=t0, x=x0, os=os)   # scalar
        drifts.append(s1 - s2)

    drifts = np.asarray(drifts)  # shape (N,)

    exp = np.random.normal(0.0, np.sqrt(2)*std_noise, size=N)

    ks = ks_2samp(drifts, exp)

    threshold = np.sqrt((1/N) * -0.5 * np.log(0.5 * 1e-5))
    assert ks.pvalue > 0.05 or abs(ks.statistic) < threshold


def test_RW_torque_bias_noise_KS():
    ax = random_n_unit_vec(3)
    ax = ax.copy()
    max_torque = 4.51
    J = 0.22
    h0 = -3.1
    h_max = 3.8

    # Bias + Noise
    biast = random_n_unit_vec(3)[1] * 0.1
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
