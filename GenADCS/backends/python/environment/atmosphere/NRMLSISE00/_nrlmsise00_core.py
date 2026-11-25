"""Full NRLMSISE-00 numerical core, transcribed from the 20041227 C model.

Adapted from the public-domain TCL Python transcription by N. Donatelli
(https://github.com/nedonatelli/TCL). Coefficients originate in the
public-domain NRL reference C tables supplied with this task.
"""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray

from ._nrlmsise00_data import PAVGM, PD, PDL, PDM, PMA, PS, PT, PTL, PTM


class NRLMSISEOutput(NamedTuple):
    """Result of the NRLMSISE-00 model.

    Attributes
    ----------
    d : ndarray
        The nine densities, indexed as the reference implementation
        orders them: d[0] He, d[1] O, d[2] N2, d[3] O2, d[4] Ar,
        d[5] total mass density, d[6] H, d[7] N, d[8] anomalous O.
        Number densities are per cubic meter and d[5] is kg/m^3 (the
        SI switch of the reference is always on here, matching the
        MATLAB wrapper).
    t : ndarray
        t[0] the exospheric temperature and t[1] the temperature at
        altitude, in Kelvin.
    """

    d: NDArray[np.floating]
    t: NDArray[np.floating]


class _Flags(NamedTuple):
    """Numerical state shared by the compiled atmosphere functions."""

    switches: NDArray[np.float64]
    sw: NDArray[np.float64]
    swc: NDArray[np.float64]


@njit(cache=True)
def _make_Flags(switches: NDArray[np.float64]) -> _Flags:
    switches = np.asarray(switches)
    sw = np.zeros(24)
    swc = np.zeros(24)
    for i in range(24):
        if i != 9:
            sw[i] = 1.0 if switches[i] == 1 else 0.0
            swc[i] = 1.0 if switches[i] > 0 else 0.0
        else:
            sw[i] = switches[i]
            swc[i] = switches[i]
    return _Flags(switches, sw, swc)


class _Input(NamedTuple):
    """Numerical state shared by the compiled atmosphere functions."""

    doy: int
    sec: float
    alt: float
    g_lat: float
    g_long: float
    lst: float
    f107A: float
    f107: float
    ap: float
    ap_a: NDArray[np.float64]


@njit(cache=True)
def _make_Input(
    doy: int,
    sec: float,
    alt: float,
    g_lat: float,
    g_long: float,
    lst: float,
    f107a: float,
    f107: float,
    ap: float,
    ap_a: NDArray[np.float64] | None = None,
) -> _Input:
    f107A = f107a
    ap_a = np.asarray(ap_a)
    return _Input(doy, sec, alt, g_lat, g_long, lst, f107A, f107, ap, ap_a)


class _Model(NamedTuple):
    """Numerical state shared by the compiled atmosphere functions."""

    gsurf: NDArray[np.float64]
    re: NDArray[np.float64]
    dd: NDArray[np.float64]
    dm04: NDArray[np.float64]
    dm16: NDArray[np.float64]
    dm28: NDArray[np.float64]
    dm32: NDArray[np.float64]
    dm40: NDArray[np.float64]
    dm01: NDArray[np.float64]
    dm14: NDArray[np.float64]
    dfa: NDArray[np.float64]
    ctloc: NDArray[np.float64]
    stloc: NDArray[np.float64]
    c2tloc: NDArray[np.float64]
    s2tloc: NDArray[np.float64]
    s3tloc: NDArray[np.float64]
    c3tloc: NDArray[np.float64]
    clong: NDArray[np.float64]
    slong: NDArray[np.float64]
    apdf: NDArray[np.float64]
    meso_tn1: NDArray[np.float64]
    meso_tn2: NDArray[np.float64]
    meso_tn3: NDArray[np.float64]
    meso_tgn1: NDArray[np.float64]
    meso_tgn2: NDArray[np.float64]
    meso_tgn3: NDArray[np.float64]
    apt: NDArray[np.float64]
    plg: NDArray[np.float64]


@njit(cache=True)
def _make_Model() -> _Model:
    return _Model(
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(1),
        np.zeros(5),
        np.zeros(4),
        np.zeros(5),
        np.zeros(2),
        np.zeros(2),
        np.zeros(2),
        np.zeros(4),
        np.zeros((4, 9)),
    )


@njit(cache=True)
def _model_glatf(self: _Model, lat: float) -> None:
    """Latitude variation of gravity and effective radius."""
    dgtr = 0.0174533
    c2 = np.cos(2.0 * dgtr * lat)
    self.gsurf[0] = 980.616 * (1.0 - 0.0026373 * c2)
    self.re[0] = 2.0 * self.gsurf[0] / (3.085462e-06 + 2.27e-09 * c2) * 1e-05


@njit(cache=True)
def _model_scalh(self: _Model, alt: float, xm: float, temp: float) -> float:
    rgas = 831.4
    g = self.gsurf[0] / (1.0 + alt / self.re[0]) ** 2
    return rgas * temp / (g * xm)


@njit(cache=True)
def _model_zeta(self: _Model, zz: float, zl: float) -> float:
    return (zz - zl) * (self.re[0] + zl) / (self.re[0] + zz)


@njit(cache=True)
def _model_densm(
    self: _Model,
    alt: float,
    d0: float,
    xm: float,
    tz: float,
    zn3: Sequence[float],
    tn3: NDArray[np.float64],
    tgn3: NDArray[np.float64],
    zn2: Sequence[float],
    tn2: NDArray[np.float64],
    tgn2: NDArray[np.float64],
) -> tuple[float, float]:
    """Temperature/density profiles for the lower atmosphere.

    Returns (density-or-temperature, tz), the C double return plus
    the pointer output.
    """
    mn3 = len(zn3)
    mn2 = len(zn2)
    rgas = 831.4
    densm_tmp = d0
    if alt > zn2[0]:
        if xm == 0.0:
            return (tz, tz)
        return (d0, tz)
    z = max(zn2[mn2 - 1], alt)
    mn = mn2
    z1 = zn2[0]
    z2 = zn2[mn - 1]
    t1 = tn2[0]
    t2 = tn2[mn - 1]
    zg = _model_zeta(self, z, z1)
    zgdif = _model_zeta(self, z2, z1)
    xs = [_model_zeta(self, zn2[k], z1) / zgdif for k in range(mn)]
    ys = [1.0 / tn2[k] for k in range(mn)]
    yd1 = -tgn2[0] / (t1 * t1) * zgdif
    yd2 = -tgn2[1] / (t2 * t2) * zgdif * ((self.re[0] + z2) / (self.re[0] + z1)) ** 2
    y2out = _spline(xs, ys, yd1, yd2)
    x = zg / zgdif
    y = _splint(xs, ys, y2out, x)
    tz = 1.0 / y
    if xm != 0.0:
        glb = self.gsurf[0] / (1.0 + z1 / self.re[0]) ** 2
        gamm = xm * glb * zgdif / rgas
        yi = _splini(xs, ys, y2out, x)
        expl = gamm * yi
        expl = min(expl, 50.0)
        densm_tmp = densm_tmp * (t1 / tz) * np.exp(-expl)
    if alt > zn3[0]:
        if xm == 0.0:
            return (tz, tz)
        return (densm_tmp, tz)
    z = alt
    mn = mn3
    z1 = zn3[0]
    z2 = zn3[mn - 1]
    t1 = tn3[0]
    t2 = tn3[mn - 1]
    zg = _model_zeta(self, z, z1)
    zgdif = _model_zeta(self, z2, z1)
    xs = [_model_zeta(self, zn3[k], z1) / zgdif for k in range(mn)]
    ys = [1.0 / tn3[k] for k in range(mn)]
    yd1 = -tgn3[0] / (t1 * t1) * zgdif
    yd2 = -tgn3[1] / (t2 * t2) * zgdif * ((self.re[0] + z2) / (self.re[0] + z1)) ** 2
    y2out = _spline(xs, ys, yd1, yd2)
    x = zg / zgdif
    y = _splint(xs, ys, y2out, x)
    tz = 1.0 / y
    if xm != 0.0:
        glb = self.gsurf[0] / (1.0 + z1 / self.re[0]) ** 2
        gamm = xm * glb * zgdif / rgas
        yi = _splini(xs, ys, y2out, x)
        expl = gamm * yi
        expl = min(expl, 50.0)
        densm_tmp = densm_tmp * (t1 / tz) * np.exp(-expl)
    if xm == 0.0:
        return (tz, tz)
    return (densm_tmp, tz)


@njit(cache=True)
def _model_densu(
    self: _Model,
    alt: float,
    dlb: float,
    tinf: float,
    tlb: float,
    xm: float,
    alpha: float,
    tz: float,
    zlb: float,
    s2: float,
    zn1: Sequence[float],
    tn1: NDArray[np.float64],
    tgn1: NDArray[np.float64],
) -> tuple[float, float]:
    """Temperature/density profiles for the MSIS thermosphere.

    Returns (density-or-temperature, tz). tn1/tgn1 are mutated at
    their first entries when below the Bates joining altitude,
    exactly as the C writes through its pointers.
    """
    mn1 = len(zn1)
    rgas = 831.4
    x = 0.0
    z1 = 0.0
    t1 = 0.0
    zgdif = 0.0
    mn = 0
    xs = [0.0]
    ys = [0.0]
    y2out = [0.0]
    za = zn1[0]
    z = max(za, alt)
    zg2 = _model_zeta(self, z, zlb)
    tt = tinf - (tinf - tlb) * np.exp(-s2 * zg2)
    ta = tt
    tz = tt
    densu_temp = tz
    if alt < za:
        dta = (tinf - ta) * s2 * ((self.re[0] + zlb) / (self.re[0] + za)) ** 2
        tgn1[0] = dta
        tn1[0] = ta
        z = max(zn1[mn1 - 1], alt)
        mn = mn1
        z1 = zn1[0]
        z2 = zn1[mn - 1]
        t1 = tn1[0]
        t2 = tn1[mn - 1]
        zg = _model_zeta(self, z, z1)
        zgdif = _model_zeta(self, z2, z1)
        xs = [_model_zeta(self, zn1[k], z1) / zgdif for k in range(mn)]
        ys = [1.0 / tn1[k] for k in range(mn)]
        yd1 = -tgn1[0] / (t1 * t1) * zgdif
        yd2 = (
            -tgn1[1] / (t2 * t2) * zgdif * ((self.re[0] + z2) / (self.re[0] + z1)) ** 2
        )
        y2out = _spline(xs, ys, yd1, yd2)
        x = zg / zgdif
        y = _splint(xs, ys, y2out, x)
        tz = 1.0 / y
        densu_temp = tz
    if xm == 0:
        return (densu_temp, tz)
    glb = self.gsurf[0] / (1.0 + zlb / self.re[0]) ** 2
    gamma = xm * glb / (s2 * rgas * tinf)
    expl = np.exp(-s2 * gamma * zg2)
    expl = min(expl, 50.0)
    if tt <= 0:
        expl = 50.0
    densa = dlb * (tlb / tt) ** (1.0 + alpha + gamma) * expl
    densu_temp = densa
    if alt >= za:
        return (densu_temp, tz)
    glb = self.gsurf[0] / (1.0 + z1 / self.re[0]) ** 2
    gamm = xm * glb * zgdif / rgas
    yi = _splini(xs, ys, y2out, x)
    expl = gamm * yi
    expl = min(expl, 50.0)
    if tz <= 0:
        expl = 50.0
    densu_temp = densu_temp * (t1 / tz) ** (1.0 + alpha) * np.exp(-expl)
    return (densu_temp, tz)


@njit(cache=True)
def _model_globe_setup(self: _Model, inp: _Input, flags: _Flags) -> None:
    """Prepare input-only harmonic terms shared by all G(L) evaluations."""
    dgtr = 0.0174533
    hr = 0.2618
    c = np.sin(inp.g_lat * dgtr)
    s = np.cos(inp.g_lat * dgtr)
    self.clong[0] = np.cos(dgtr * inp.g_long)
    self.slong[0] = np.sin(dgtr * inp.g_long)
    c2 = c * c
    c4 = c2 * c2
    s2 = s * s
    plg = self.plg
    plg[0][1] = c
    plg[0][2] = 0.5 * (3.0 * c2 - 1.0)
    plg[0][3] = 0.5 * (5.0 * c * c2 - 3.0 * c)
    plg[0][4] = (35.0 * c4 - 30.0 * c2 + 3.0) / 8.0
    plg[0][5] = (63.0 * c2 * c2 * c - 70.0 * c2 * c + 15.0 * c) / 8.0
    plg[0][6] = (11.0 * c * plg[0][5] - 5.0 * plg[0][4]) / 6.0
    plg[1][1] = s
    plg[1][2] = 3.0 * c * s
    plg[1][3] = 1.5 * (5.0 * c2 - 1.0) * s
    plg[1][4] = 2.5 * (7.0 * c2 * c - 3.0 * c) * s
    plg[1][5] = 1.875 * (21.0 * c4 - 14.0 * c2 + 1.0) * s
    plg[1][6] = (11.0 * c * plg[1][5] - 6.0 * plg[1][4]) / 5.0
    plg[2][2] = 3.0 * s2
    plg[2][3] = 15.0 * s2 * c
    plg[2][4] = 7.5 * (7.0 * c2 - 1.0) * s2
    plg[2][5] = 3.0 * c * plg[2][4] - 2.0 * plg[2][3]
    plg[2][6] = (11.0 * c * plg[2][5] - 7.0 * plg[2][4]) / 4.0
    plg[2][7] = (13.0 * c * plg[2][6] - 8.0 * plg[2][5]) / 5.0
    plg[3][3] = 15.0 * s2 * s
    plg[3][4] = 105.0 * s2 * s * c
    plg[3][5] = (9.0 * c * plg[3][4] - 7.0 * plg[3][3]) / 2.0
    plg[3][6] = (11.0 * c * plg[3][5] - 8.0 * plg[3][4]) / 3.0
    if not ((flags.sw[7] == 0 and flags.sw[8] == 0) and flags.sw[14] == 0):
        tloc = inp.lst
        self.stloc[0] = np.sin(hr * tloc)
        self.ctloc[0] = np.cos(hr * tloc)
        self.s2tloc[0] = np.sin(2.0 * hr * tloc)
        self.c2tloc[0] = np.cos(2.0 * hr * tloc)
        self.s3tloc[0] = np.sin(3.0 * hr * tloc)
        self.c3tloc[0] = np.cos(3.0 * hr * tloc)


@njit(cache=True)
def _model_globe7(
    self: _Model, p: NDArray[np.float64], inp: _Input, flags: _Flags
) -> float:
    """Upper-thermosphere G(L); mutates the LPOLY shared state."""
    t = [0.0] * 15
    sr = 7.2722e-05
    dgtr = 0.0174533
    dr = 0.0172142
    hr = 0.2618
    tloc = inp.lst
    plg = self.plg
    cd32 = np.cos(dr * (inp.doy - p[31]))
    cd18 = np.cos(2.0 * dr * (inp.doy - p[17]))
    cd14 = np.cos(dr * (inp.doy - p[13]))
    cd39 = np.cos(2.0 * dr * (inp.doy - p[38]))
    df = inp.f107 - inp.f107A
    self.dfa[0] = inp.f107A - 150.0
    dfa = self.dfa[0]
    t[0] = (
        p[19] * df * (1.0 + p[59] * dfa)
        + p[20] * df * df
        + p[21] * dfa
        + p[29] * dfa**2
    )
    f1 = 1.0 + (p[47] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1]
    f2 = 1.0 + (p[49] * dfa + p[19] * df + p[20] * df * df) * flags.swc[1]
    t[1] = (
        p[1] * plg[0][2]
        + p[2] * plg[0][4]
        + p[22] * plg[0][6]
        + p[14] * plg[0][2] * dfa * flags.swc[1]
        + p[26] * plg[0][1]
    )
    t[2] = p[18] * cd32
    t[3] = (p[15] + p[16] * plg[0][2]) * cd18
    t[4] = f1 * (p[9] * plg[0][1] + p[10] * plg[0][3]) * cd14
    t[5] = p[37] * plg[0][1] * cd39
    if flags.sw[7]:
        t71 = p[11] * plg[1][2] * cd14 * flags.swc[5]
        t72 = p[12] * plg[1][2] * cd14 * flags.swc[5]
        t[6] = f2 * (
            (p[3] * plg[1][1] + p[4] * plg[1][3] + p[27] * plg[1][5] + t71)
            * self.ctloc[0]
            + (p[6] * plg[1][1] + p[7] * plg[1][3] + p[28] * plg[1][5] + t72)
            * self.stloc[0]
        )
    if flags.sw[8]:
        t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5]
        t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5]
        t[7] = f2 * (
            (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc[0]
            + (p[8] * plg[2][2] + p[42] * plg[2][4] + t82) * self.s2tloc[0]
        )
    if flags.sw[14]:
        t[13] = f2 * (
            (
                p[39] * plg[3][3]
                + (p[93] * plg[3][4] + p[46] * plg[3][6]) * cd14 * flags.swc[5]
            )
            * self.s3tloc[0]
            + (
                p[40] * plg[3][3]
                + (p[94] * plg[3][4] + p[48] * plg[3][6]) * cd14 * flags.swc[5]
            )
            * self.c3tloc[0]
        )
    if flags.sw[9] == -1:
        ap = inp.ap_a
        if p[51] != 0:
            exp1 = np.exp(
                -10800.0
                * np.sqrt(p[51] * p[51])
                / (1.0 + p[138] * (45.0 - np.sqrt(inp.g_lat * inp.g_lat)))
            )
            exp1 = min(exp1, 0.99999)
            if p[24] < 0.0001:
                pass
            self.apt[0] = _sg0(exp1, p, ap)
            if flags.sw[9]:
                t[8] = self.apt[0] * (
                    p[50]
                    + p[96] * plg[0][2]
                    + p[54] * plg[0][4]
                    + (p[125] * plg[0][1] + p[126] * plg[0][3] + p[127] * plg[0][5])
                    * cd14
                    * flags.swc[5]
                    + (p[128] * plg[1][1] + p[129] * plg[1][3] + p[130] * plg[1][5])
                    * flags.swc[7]
                    * np.cos(hr * (tloc - p[131]))
                )
    else:
        apd = inp.ap - 4.0
        p44 = p[43]
        p45 = p[44]
        if p44 < 0:
            p44 = 1e-05
        self.apdf[0] = apd + (p45 - 1.0) * (apd + (np.exp(-p44 * apd) - 1.0) / p44)
        if flags.sw[9]:
            t[8] = self.apdf[0] * (
                p[32]
                + p[45] * plg[0][2]
                + p[34] * plg[0][4]
                + (p[100] * plg[0][1] + p[101] * plg[0][3] + p[102] * plg[0][5])
                * cd14
                * flags.swc[5]
                + (p[121] * plg[1][1] + p[122] * plg[1][3] + p[123] * plg[1][5])
                * flags.swc[7]
                * np.cos(hr * (tloc - p[124]))
            )
    if flags.sw[10] and inp.g_long > -1000.0:
        if flags.sw[11]:
            t[10] = (1.0 + p[80] * dfa * flags.swc[1]) * (
                (
                    p[64] * plg[1][2]
                    + p[65] * plg[1][4]
                    + p[66] * plg[1][6]
                    + p[103] * plg[1][1]
                    + p[104] * plg[1][3]
                    + p[105] * plg[1][5]
                    + flags.swc[5]
                    * (p[109] * plg[1][1] + p[110] * plg[1][3] + p[111] * plg[1][5])
                    * cd14
                )
                * self.clong[0]
                + (
                    p[90] * plg[1][2]
                    + p[91] * plg[1][4]
                    + p[92] * plg[1][6]
                    + p[106] * plg[1][1]
                    + p[107] * plg[1][3]
                    + p[108] * plg[1][5]
                    + flags.swc[5]
                    * (p[112] * plg[1][1] + p[113] * plg[1][3] + p[114] * plg[1][5])
                    * cd14
                )
                * self.slong[0]
            )
        if flags.sw[12]:
            t[11] = (
                (1.0 + p[95] * plg[0][1])
                * (1.0 + p[81] * dfa * flags.swc[1])
                * (1.0 + p[119] * plg[0][1] * flags.swc[5] * cd14)
                * (
                    (p[68] * plg[0][1] + p[69] * plg[0][3] + p[70] * plg[0][5])
                    * np.cos(sr * (inp.sec - p[71]))
                )
            )
            t[11] += (
                flags.swc[11]
                * (p[76] * plg[2][3] + p[77] * plg[2][5] + p[78] * plg[2][7])
                * np.cos(sr * (inp.sec - p[79]) + 2.0 * dgtr * inp.g_long)
                * (1.0 + p[137] * dfa * flags.swc[1])
            )
        if flags.sw[13]:
            if flags.sw[9] == -1:
                if p[51]:
                    t[12] = (
                        self.apt[0]
                        * flags.swc[11]
                        * (1.0 + p[132] * plg[0][1])
                        * (
                            (p[52] * plg[1][2] + p[98] * plg[1][4] + p[67] * plg[1][6])
                            * np.cos(dgtr * (inp.g_long - p[97]))
                        )
                        + self.apt[0]
                        * flags.swc[11]
                        * flags.swc[5]
                        * (p[133] * plg[1][1] + p[134] * plg[1][3] + p[135] * plg[1][5])
                        * cd14
                        * np.cos(dgtr * (inp.g_long - p[136]))
                        + self.apt[0]
                        * flags.swc[12]
                        * (p[55] * plg[0][1] + p[56] * plg[0][3] + p[57] * plg[0][5])
                        * np.cos(sr * (inp.sec - p[58]))
                    )
            else:
                t[12] = (
                    self.apdf[0]
                    * flags.swc[11]
                    * (1.0 + p[120] * plg[0][1])
                    * (
                        (p[60] * plg[1][2] + p[61] * plg[1][4] + p[62] * plg[1][6])
                        * np.cos(dgtr * (inp.g_long - p[63]))
                    )
                    + self.apdf[0]
                    * flags.swc[11]
                    * flags.swc[5]
                    * (p[115] * plg[1][1] + p[116] * plg[1][3] + p[117] * plg[1][5])
                    * cd14
                    * np.cos(dgtr * (inp.g_long - p[118]))
                    + self.apdf[0]
                    * flags.swc[12]
                    * (p[83] * plg[0][1] + p[84] * plg[0][3] + p[85] * plg[0][5])
                    * np.cos(sr * (inp.sec - p[75]))
                )
    tinf = p[30]
    for i in range(14):
        tinf = tinf + abs(flags.sw[i + 1]) * t[i]
    return tinf


@njit(cache=True)
def _model_glob7s(
    self: _Model, p: NDArray[np.float64], inp: _Input, flags: _Flags
) -> float:
    """Lower-atmosphere version of the G(L) function."""
    pset = 2.0
    t = [0.0] * 14
    dr = 0.0172142
    if p[99] != 0 and p[99] != pset:
        raise ValueError("Wrong parameter set for glob7s")
    cd32 = np.cos(dr * (inp.doy - p[31]))
    cd18 = np.cos(2.0 * dr * (inp.doy - p[17]))
    cd14 = np.cos(dr * (inp.doy - p[13]))
    cd39 = np.cos(2.0 * dr * (inp.doy - p[38]))
    plg = self.plg
    t[0] = p[21] * self.dfa[0]
    t[1] = (
        p[1] * plg[0][2]
        + p[2] * plg[0][4]
        + p[22] * plg[0][6]
        + p[26] * plg[0][1]
        + p[14] * plg[0][3]
        + p[59] * plg[0][5]
    )
    t[2] = (p[18] + p[47] * plg[0][2] + p[29] * plg[0][4]) * cd32
    t[3] = (p[15] + p[16] * plg[0][2] + p[30] * plg[0][4]) * cd18
    t[4] = (p[9] * plg[0][1] + p[10] * plg[0][3] + p[20] * plg[0][5]) * cd14
    t[5] = p[37] * plg[0][1] * cd39
    if flags.sw[7]:
        t71 = p[11] * plg[1][2] * cd14 * flags.swc[5]
        t72 = p[12] * plg[1][2] * cd14 * flags.swc[5]
        t[6] = (p[3] * plg[1][1] + p[4] * plg[1][3] + t71) * self.ctloc[0] + (
            p[6] * plg[1][1] + p[7] * plg[1][3] + t72
        ) * self.stloc[0]
    if flags.sw[8]:
        t81 = (p[23] * plg[2][3] + p[35] * plg[2][5]) * cd14 * flags.swc[5]
        t82 = (p[33] * plg[2][3] + p[36] * plg[2][5]) * cd14 * flags.swc[5]
        t[7] = (p[5] * plg[2][2] + p[41] * plg[2][4] + t81) * self.c2tloc[0] + (
            p[8] * plg[2][2] + p[42] * plg[2][4] + t82
        ) * self.s2tloc[0]
    if flags.sw[14]:
        t[13] = p[39] * plg[3][3] * self.s3tloc[0] + p[40] * plg[3][3] * self.c3tloc[0]
    if flags.sw[9]:
        if flags.sw[9] == 1:
            t[8] = self.apdf[0] * (p[32] + p[45] * plg[0][2] * flags.swc[2])
        if flags.sw[9] == -1:
            t[8] = p[50] * self.apt[0] + p[96] * plg[0][2] * self.apt[0] * flags.swc[2]
    if not (flags.sw[10] == 0 or flags.sw[11] == 0 or inp.g_long <= -1000.0):
        t[10] = (
            1.0
            + plg[0][1]
            * (
                p[80] * flags.swc[5] * np.cos(dr * (inp.doy - p[81]))
                + p[85] * flags.swc[6] * np.cos(2.0 * dr * (inp.doy - p[86]))
            )
            + p[83] * flags.swc[3] * np.cos(dr * (inp.doy - p[84]))
            + p[87] * flags.swc[4] * np.cos(2.0 * dr * (inp.doy - p[88]))
        ) * (
            (
                p[64] * plg[1][2]
                + p[65] * plg[1][4]
                + p[66] * plg[1][6]
                + p[74] * plg[1][1]
                + p[75] * plg[1][3]
                + p[76] * plg[1][5]
            )
            * self.clong[0]
            + (
                p[90] * plg[1][2]
                + p[91] * plg[1][4]
                + p[92] * plg[1][6]
                + p[77] * plg[1][1]
                + p[78] * plg[1][3]
                + p[79] * plg[1][5]
            )
            * self.slong[0]
        )
    tt = 0.0
    for i in range(14):
        tt += abs(flags.sw[i + 1]) * t[i]
    return tt


@njit(cache=True)
def _model_gts7(self: _Model, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
    """Thermospheric portion (alt > 72.5 km)."""
    zn1 = [120.0, 110.0, 100.0, 90.0, 72.5]
    dgtr = 0.0174533
    dr = 0.0172142
    alpha = [-0.38, 0.0, 0.0, 0.0, 0.17, 0.0, -0.38, 0.0, 0.0]
    altl = [200.0, 300.0, 160.0, 250.0, 240.0, 450.0, 320.0, 450.0]
    d = [0.0] * 9
    t = [0.0, 0.0]
    za = PDL[1][15]
    zn1[0] = za
    if inp.alt > zn1[0]:
        tinf = (
            PTM[0] * PT[0] * (1.0 + flags.sw[16] * _model_globe7(self, PT, inp, flags))
        )
    else:
        tinf = PTM[0] * PT[0]
    t[0] = tinf
    if inp.alt > zn1[4]:
        g0v = (
            PTM[3] * PS[0] * (1.0 + flags.sw[19] * _model_globe7(self, PS, inp, flags))
        )
    else:
        g0v = PTM[3] * PS[0]
    tlb = (
        PTM[1]
        * (1.0 + flags.sw[17] * _model_globe7(self, PD[3], inp, flags))
        * PD[3][0]
    )
    s = g0v / (tinf - tlb)
    if inp.alt < 300.0:
        self.meso_tn1[1] = (
            PTM[6]
            * PTL[0][0]
            / (1.0 - flags.sw[18] * _model_glob7s(self, PTL[0], inp, flags))
        )
        self.meso_tn1[2] = (
            PTM[2]
            * PTL[1][0]
            / (1.0 - flags.sw[18] * _model_glob7s(self, PTL[1], inp, flags))
        )
        self.meso_tn1[3] = (
            PTM[7]
            * PTL[2][0]
            / (1.0 - flags.sw[18] * _model_glob7s(self, PTL[2], inp, flags))
        )
        self.meso_tn1[4] = (
            PTM[4]
            * PTL[3][0]
            / (
                1.0
                - flags.sw[18] * flags.sw[20] * _model_glob7s(self, PTL[3], inp, flags)
            )
        )
        self.meso_tgn1[1] = (
            PTM[8]
            * PMA[8][0]
            * (
                1.0
                + flags.sw[18] * flags.sw[20] * _model_glob7s(self, PMA[8], inp, flags)
            )
            * self.meso_tn1[4]
            * self.meso_tn1[4]
            / (PTM[4] * PTL[3][0]) ** 2
        )
    else:
        self.meso_tn1[1] = PTM[6] * PTL[0][0]
        self.meso_tn1[2] = PTM[2] * PTL[1][0]
        self.meso_tn1[3] = PTM[7] * PTL[2][0]
        self.meso_tn1[4] = PTM[4] * PTL[3][0]
        self.meso_tgn1[1] = (
            PTM[8]
            * PMA[8][0]
            * self.meso_tn1[4]
            * self.meso_tn1[4]
            / (PTM[4] * PTL[3][0]) ** 2
        )
    g28 = flags.sw[21] * _model_globe7(self, PD[2], inp, flags)
    zhf = PDL[1][24] * (
        1.0
        + flags.sw[5]
        * PDL[0][24]
        * np.sin(dgtr * inp.g_lat)
        * np.cos(dr * (inp.doy - PT[13]))
    )
    t[0] = tinf
    xmm = PDM[2][4]
    z = inp.alt
    db28 = PDM[2][0] * np.exp(g28) * PD[2][0]
    d[2], t[1] = _model_densu(
        self,
        z,
        db28,
        tinf,
        tlb,
        28.0,
        alpha[2],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    zh28 = PDM[2][2] * zhf
    zhm28 = PDM[2][3] * PDL[1][5]
    xmd = 28.0 - xmm
    b28, tz = _model_densu(
        self,
        zh28,
        db28,
        tinf,
        tlb,
        xmd,
        alpha[2] - 1.0,
        0.0,
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z <= altl[2]:
        self.dm28[0], tz = _model_densu(
            self,
            z,
            b28,
            tinf,
            tlb,
            xmm,
            alpha[2],
            tz,
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        d[2] = _dnet(d[2], self.dm28[0], zhm28, xmm, 28.0)
    g4 = flags.sw[21] * _model_globe7(self, PD[0], inp, flags)
    db04 = PDM[0][0] * np.exp(g4) * PD[0][0]
    d[0], t[1] = _model_densu(
        self,
        z,
        db04,
        tinf,
        tlb,
        4.0,
        alpha[0],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z < altl[0]:
        zh04 = PDM[0][2]
        b04, t[1] = _model_densu(
            self,
            zh04,
            db04,
            tinf,
            tlb,
            4.0 - xmm,
            alpha[0] - 1.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        self.dm04[0], t[1] = _model_densu(
            self,
            z,
            b04,
            tinf,
            tlb,
            xmm,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zhm04 = zhm28
        d[0] = _dnet(d[0], self.dm04[0], zhm04, xmm, 4.0)
        rl = np.log(b28 * PDM[0][1] / b04)
        zc04 = PDM[0][4] * PDL[1][0]
        hc04 = PDM[0][5] * PDL[1][1]
        d[0] = d[0] * _ccor(z, rl, hc04, zc04)
    g16 = flags.sw[21] * _model_globe7(self, PD[1], inp, flags)
    db16 = PDM[1][0] * np.exp(g16) * PD[1][0]
    d[1], t[1] = _model_densu(
        self,
        z,
        db16,
        tinf,
        tlb,
        16.0,
        alpha[1],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z <= altl[1]:
        zh16 = PDM[1][2]
        b16, t[1] = _model_densu(
            self,
            zh16,
            db16,
            tinf,
            tlb,
            16.0 - xmm,
            alpha[1] - 1.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        self.dm16[0], t[1] = _model_densu(
            self,
            z,
            b16,
            tinf,
            tlb,
            xmm,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zhm16 = zhm28
        d[1] = _dnet(d[1], self.dm16[0], zhm16, xmm, 16.0)
        rl = (
            PDM[1][1]
            * PDL[1][16]
            * (1.0 + flags.sw[1] * PDL[0][23] * (inp.f107A - 150.0))
        )
        hc16 = PDM[1][5] * PDL[1][3]
        zc16 = PDM[1][4] * PDL[1][2]
        hc216 = PDM[1][5] * PDL[1][4]
        d[1] = d[1] * _ccor2(z, rl, hc16, zc16, hc216)
        hcc16 = PDM[1][7] * PDL[1][13]
        zcc16 = PDM[1][6] * PDL[1][12]
        rc16 = PDM[1][3] * PDL[1][14]
        d[1] = d[1] * _ccor(z, rc16, hcc16, zcc16)
    g32 = flags.sw[21] * _model_globe7(self, PD[4], inp, flags)
    db32 = PDM[3][0] * np.exp(g32) * PD[4][0]
    d[3], t[1] = _model_densu(
        self,
        z,
        db32,
        tinf,
        tlb,
        32.0,
        alpha[3],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15]:
        if z <= altl[3]:
            zh32 = PDM[3][2]
            b32, t[1] = _model_densu(
                self,
                zh32,
                db32,
                tinf,
                tlb,
                32.0 - xmm,
                alpha[3] - 1.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            self.dm32[0], t[1] = _model_densu(
                self,
                z,
                b32,
                tinf,
                tlb,
                xmm,
                0.0,
                t[1],
                PTM[5],
                s,
                zn1,
                self.meso_tn1,
                self.meso_tgn1,
            )
            zhm32 = zhm28
            d[3] = _dnet(d[3], self.dm32[0], zhm32, xmm, 32.0)
            rl = np.log(b28 * PDM[3][1] / b32)
            hc32 = PDM[3][5] * PDL[1][7]
            zc32 = PDM[3][4] * PDL[1][6]
            d[3] = d[3] * _ccor(z, rl, hc32, zc32)
        hcc32 = PDM[3][7] * PDL[1][22]
        hcc232 = PDM[3][7] * PDL[0][22]
        zcc32 = PDM[3][6] * PDL[1][21]
        rc32 = (
            PDM[3][3]
            * PDL[1][23]
            * (1.0 + flags.sw[1] * PDL[0][23] * (inp.f107A - 150.0))
        )
        d[3] = d[3] * _ccor2(z, rc32, hcc32, zcc32, hcc232)
    g40 = flags.sw[21] * _model_globe7(self, PD[5], inp, flags)
    db40 = PDM[4][0] * np.exp(g40) * PD[5][0]
    d[4], t[1] = _model_densu(
        self,
        z,
        db40,
        tinf,
        tlb,
        40.0,
        alpha[4],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z <= altl[4]:
        zh40 = PDM[4][2]
        b40, t[1] = _model_densu(
            self,
            zh40,
            db40,
            tinf,
            tlb,
            40.0 - xmm,
            alpha[4] - 1.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        self.dm40[0], t[1] = _model_densu(
            self,
            z,
            b40,
            tinf,
            tlb,
            xmm,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zhm40 = zhm28
        d[4] = _dnet(d[4], self.dm40[0], zhm40, xmm, 40.0)
        rl = np.log(b28 * PDM[4][1] / b40)
        hc40 = PDM[4][5] * PDL[1][9]
        zc40 = PDM[4][4] * PDL[1][8]
        d[4] = d[4] * _ccor(z, rl, hc40, zc40)
    g1 = flags.sw[21] * _model_globe7(self, PD[6], inp, flags)
    db01 = PDM[5][0] * np.exp(g1) * PD[6][0]
    d[6], t[1] = _model_densu(
        self,
        z,
        db01,
        tinf,
        tlb,
        1.0,
        alpha[6],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z <= altl[6]:
        zh01 = PDM[5][2]
        b01, t[1] = _model_densu(
            self,
            zh01,
            db01,
            tinf,
            tlb,
            1.0 - xmm,
            alpha[6] - 1.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        self.dm01[0], t[1] = _model_densu(
            self,
            z,
            b01,
            tinf,
            tlb,
            xmm,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zhm01 = zhm28
        d[6] = _dnet(d[6], self.dm01[0], zhm01, xmm, 1.0)
        rl = np.log(b28 * PDM[5][1] * np.sqrt(PDL[1][17] * PDL[1][17]) / b01)
        hc01 = PDM[5][5] * PDL[1][11]
        zc01 = PDM[5][4] * PDL[1][10]
        d[6] = d[6] * _ccor(z, rl, hc01, zc01)
        hcc01 = PDM[5][7] * PDL[1][19]
        zcc01 = PDM[5][6] * PDL[1][18]
        rc01 = PDM[5][3] * PDL[1][20]
        d[6] = d[6] * _ccor(z, rc01, hcc01, zcc01)
    g14 = flags.sw[21] * _model_globe7(self, PD[7], inp, flags)
    db14 = PDM[6][0] * np.exp(g14) * PD[7][0]
    d[7], t[1] = _model_densu(
        self,
        z,
        db14,
        tinf,
        tlb,
        14.0,
        alpha[7],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[15] and z <= altl[7]:
        zh14 = PDM[6][2]
        b14, t[1] = _model_densu(
            self,
            zh14,
            db14,
            tinf,
            tlb,
            14.0 - xmm,
            alpha[7] - 1.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        self.dm14[0], t[1] = _model_densu(
            self,
            z,
            b14,
            tinf,
            tlb,
            xmm,
            0.0,
            t[1],
            PTM[5],
            s,
            zn1,
            self.meso_tn1,
            self.meso_tgn1,
        )
        zhm14 = zhm28
        d[7] = _dnet(d[7], self.dm14[0], zhm14, xmm, 14.0)
        rl = np.log(b28 * PDM[6][1] * np.sqrt(PDL[0][2] * PDL[0][2]) / b14)
        hc14 = PDM[6][5] * PDL[0][1]
        zc14 = PDM[6][4] * PDL[0][0]
        d[7] = d[7] * _ccor(z, rl, hc14, zc14)
        hcc14 = PDM[6][7] * PDL[0][4]
        zcc14 = PDM[6][6] * PDL[0][3]
        rc14 = PDM[6][3] * PDL[0][5]
        d[7] = d[7] * _ccor(z, rc14, hcc14, zcc14)
    g16h = flags.sw[21] * _model_globe7(self, PD[8], inp, flags)
    db16h = PDM[7][0] * np.exp(g16h) * PD[8][0]
    tho = PDM[7][9] * PDL[0][6]
    dde, t[1] = _model_densu(
        self,
        z,
        db16h,
        tho,
        tho,
        16.0,
        alpha[8],
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    zsht = PDM[7][5]
    zmho = PDM[7][4]
    zsho = _model_scalh(self, zmho, 16.0, tho)
    d[8] = dde * np.exp(-zsht / zsho * (np.exp(-(z - zmho) / zsht) - 1.0))
    d[5] = 1.66e-24 * (
        4.0 * d[0]
        + 16.0 * d[1]
        + 28.0 * d[2]
        + 32.0 * d[3]
        + 40.0 * d[4]
        + d[6]
        + 14.0 * d[7]
    )
    z = np.sqrt(inp.alt * inp.alt)
    _, t[1] = _model_densu(
        self,
        z,
        1.0,
        tinf,
        tlb,
        0.0,
        0.0,
        t[1],
        PTM[5],
        s,
        zn1,
        self.meso_tn1,
        self.meso_tgn1,
    )
    if flags.sw[0]:
        for i in range(9):
            d[i] = d[i] * 1000000.0
        d[5] = d[5] / 1000.0
    return NRLMSISEOutput(np.array(d), np.array(t))


@njit(cache=True)
def _modelgtd7(self: _Model, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
    """The full model: thermosphere plus lower atmosphere."""
    zn3 = [32.5, 20.0, 15.0, 10.0, 0.0]
    zn2 = [72.5, 55.0, 45.0, 32.5]
    zmix = 62.5
    _model_globe_setup(self, inp, flags)
    xlat = inp.g_lat
    if flags.sw[2] == 0:
        xlat = 45.0
    _model_glatf(self, xlat)
    xmm = PDM[2][4]
    altt = max(zn2[0], inp.alt)
    upper_input = _Input(
        inp.doy,
        inp.sec,
        altt,
        inp.g_lat,
        inp.g_long,
        inp.lst,
        inp.f107A,
        inp.f107,
        inp.ap,
        inp.ap_a,
    )
    soutput = _model_gts7(self, upper_input, flags)
    if flags.sw[0]:
        dm28m = self.dm28[0] * 1000000.0
    else:
        dm28m = self.dm28[0]
    d = [0.0] * 9
    t = [soutput.t[0], soutput.t[1]]
    if inp.alt >= zn2[0]:
        return soutput
    self.meso_tgn2[0] = self.meso_tgn1[1]
    self.meso_tn2[0] = self.meso_tn1[4]
    self.meso_tn2[1] = (
        PMA[0][0]
        * PAVGM[0]
        / (1.0 - flags.sw[20] * _model_glob7s(self, PMA[0], inp, flags))
    )
    self.meso_tn2[2] = (
        PMA[1][0]
        * PAVGM[1]
        / (1.0 - flags.sw[20] * _model_glob7s(self, PMA[1], inp, flags))
    )
    self.meso_tn2[3] = (
        PMA[2][0]
        * PAVGM[2]
        / (1.0 - flags.sw[20] * flags.sw[22] * _model_glob7s(self, PMA[2], inp, flags))
    )
    self.meso_tgn2[1] = (
        PAVGM[8]
        * PMA[9][0]
        * (1.0 + flags.sw[20] * flags.sw[22] * _model_glob7s(self, PMA[9], inp, flags))
        * self.meso_tn2[3]
        * self.meso_tn2[3]
        / (PMA[2][0] * PAVGM[2]) ** 2
    )
    self.meso_tn3[0] = self.meso_tn2[3]
    if inp.alt <= zn3[0]:
        self.meso_tgn3[0] = self.meso_tgn2[1]
        self.meso_tn3[1] = (
            PMA[3][0]
            * PAVGM[3]
            / (1.0 - flags.sw[22] * _model_glob7s(self, PMA[3], inp, flags))
        )
        self.meso_tn3[2] = (
            PMA[4][0]
            * PAVGM[4]
            / (1.0 - flags.sw[22] * _model_glob7s(self, PMA[4], inp, flags))
        )
        self.meso_tn3[3] = (
            PMA[5][0]
            * PAVGM[5]
            / (1.0 - flags.sw[22] * _model_glob7s(self, PMA[5], inp, flags))
        )
        self.meso_tn3[4] = (
            PMA[6][0]
            * PAVGM[6]
            / (1.0 - flags.sw[22] * _model_glob7s(self, PMA[6], inp, flags))
        )
        self.meso_tgn3[1] = (
            PMA[7][0]
            * PAVGM[7]
            * (1.0 + flags.sw[22] * _model_glob7s(self, PMA[7], inp, flags))
            * self.meso_tn3[4]
            * self.meso_tn3[4]
            / (PMA[6][0] * PAVGM[6]) ** 2
        )
    dmc = 0.0
    if inp.alt > zmix:
        dmc = 1.0 - (zn2[0] - inp.alt) / (zn2[0] - zmix)
    dz28 = soutput.d[2]
    dmr = soutput.d[2] / dm28m - 1.0
    tz = 0.0
    d[2], tz = _model_densm(
        self,
        inp.alt,
        dm28m,
        xmm,
        tz,
        zn3,
        self.meso_tn3,
        self.meso_tgn3,
        zn2,
        self.meso_tn2,
        self.meso_tgn2,
    )
    d[2] = d[2] * (1.0 + dmr * dmc)
    dmr = soutput.d[0] / (dz28 * PDM[0][1]) - 1.0
    d[0] = d[2] * PDM[0][1] * (1.0 + dmr * dmc)
    d[1] = 0.0
    d[8] = 0.0
    dmr = soutput.d[3] / (dz28 * PDM[3][1]) - 1.0
    d[3] = d[2] * PDM[3][1] * (1.0 + dmr * dmc)
    dmr = soutput.d[4] / (dz28 * PDM[4][1]) - 1.0
    d[4] = d[2] * PDM[4][1] * (1.0 + dmr * dmc)
    d[6] = 0.0
    d[7] = 0.0
    d[5] = 1.66e-24 * (
        4.0 * d[0]
        + 16.0 * d[1]
        + 28.0 * d[2]
        + 32.0 * d[3]
        + 40.0 * d[4]
        + d[6]
        + 14.0 * d[7]
    )
    if flags.sw[0]:
        d[5] = d[5] / 1000.0
    self.dd[0], tz = _model_densm(
        self,
        inp.alt,
        1.0,
        0.0,
        tz,
        zn3,
        self.meso_tn3,
        self.meso_tgn3,
        zn2,
        self.meso_tn2,
        self.meso_tgn2,
    )
    t[1] = tz
    return NRLMSISEOutput(np.array(d), np.array(t))


@njit(cache=True)
def _modelgtd7d(self: _Model, inp: _Input, flags: _Flags) -> NRLMSISEOutput:
    """The full model with anomalous O in the effective density."""
    out = _modelgtd7(self, inp, flags)
    out.d[5] = 1.66e-24 * (
        4.0 * out.d[0]
        + 16.0 * out.d[1]
        + 28.0 * out.d[2]
        + 32.0 * out.d[3]
        + 40.0 * out.d[4]
        + out.d[6]
        + 14.0 * out.d[7]
        + 16.0 * out.d[8]
    )
    if flags.sw[0]:
        out.d[5] = out.d[5] / 1000.0
    return out


@njit(cache=True)
def _ccor(alt: float, r: float, h1: float, zh: float) -> float:
    """Chemistry/dissociation correction."""
    e = (alt - zh) / h1
    if e > 70:
        return np.exp(0)
    if e < -70:
        return np.exp(r)
    ex = np.exp(e)
    e = r / (1.0 + ex)
    return np.exp(e)


@njit(cache=True)
def _ccor2(alt: float, r: float, h1: float, zh: float, h2: float) -> float:
    """Chemistry/dissociation correction with two scale lengths."""
    e1 = (alt - zh) / h1
    e2 = (alt - zh) / h2
    if e1 > 70 or e2 > 70:
        return np.exp(0)
    if e1 < -70 and e2 < -70:
        return np.exp(r)
    ex1 = np.exp(e1)
    ex2 = np.exp(e2)
    ccor2v = r / (1.0 + 0.5 * (ex1 + ex2))
    return np.exp(ccor2v)


@njit(cache=True)
def _dnet(dde: float, dm: float, zhm: float, xmm: float, xm: float) -> float:
    """Turbopause correction (root mean density)."""
    a = zhm / (xmm - xm)
    if not (dm > 0 and dde > 0):
        if dde == 0 and dm == 0:
            dde = 1.0
        if dm == 0:
            return dde
        if dde == 0:
            return dm
    ylog = a * np.log(dm / dde)
    if ylog < -10:
        return dde
    if ylog > 10:
        return dm
    return dde * (1.0 + np.exp(ylog)) ** (1.0 / a)


@njit(cache=True)
def _splini(
    xa: Sequence[float], ya: Sequence[float], y2a: Sequence[float], x: float
) -> float:
    """Integrate the cubic spline from xa[0] to x."""
    n = len(xa)
    yi = 0.0
    klo = 0
    khi = 1
    while x > xa[klo] and khi < n:
        xx = x
        if khi < n - 1:
            xx = min(xa[khi], x)
        h = xa[khi] - xa[klo]
        a = (xa[khi] - xx) / h
        b = (xx - xa[klo]) / h
        a2 = a * a
        b2 = b * b
        yi += (
            (1.0 - a2) * ya[klo] / 2.0
            + b2 * ya[khi] / 2.0
            + (
                (-(1.0 + a2 * a2) / 4.0 + a2 / 2.0) * y2a[klo]
                + (b2 * b2 / 4.0 - b2 / 2.0) * y2a[khi]
            )
            * h
            * h
            / 6.0
        ) * h
        klo += 1
        khi += 1
    return yi


@njit(cache=True)
def _splint(
    xa: Sequence[float], ya: Sequence[float], y2a: Sequence[float], x: float
) -> float:
    """Cubic spline interpolation (Numerical Recipes form)."""
    n = len(xa)
    klo = 0
    khi = n - 1
    while khi - klo > 1:
        k = (khi + klo) // 2
        if xa[k] > x:
            khi = k
        else:
            klo = k
    h = xa[khi] - xa[klo]
    if h == 0.0:
        return np.nan
    a = (xa[khi] - x) / h
    b = (x - xa[klo]) / h
    return (
        a * ya[klo]
        + b * ya[khi]
        + ((a * a * a - a) * y2a[klo] + (b * b * b - b) * y2a[khi]) * h * h / 6.0
    )


@njit(cache=True)
def _spline(
    x: Sequence[float], y: Sequence[float], yp1: float, ypn: float
) -> list[float]:
    """Second derivatives of the cubic spline (Numerical Recipes)."""
    n = len(x)
    y2 = [0.0] * n
    u = [0.0] * n
    if yp1 > 9.9e29:
        y2[0] = 0.0
        u[0] = 0.0
    else:
        y2[0] = -0.5
        u[0] = 3.0 / (x[1] - x[0]) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (
            6.0
            * (
                (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1])
            )
            / (x[i + 1] - x[i - 1])
            - sig * u[i - 1]
        ) / p
    if ypn > 9.9e29:
        qn = 0.0
        un = 0.0
    else:
        qn = 0.5
        un = (
            3.0
            / (x[n - 1] - x[n - 2])
            * (ypn - (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]))
        )
    y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


@njit(cache=True)
def _g0(a: float, p: NDArray[np.float64]) -> float:
    """3-hour magnetic activity function, Eq. A24d."""
    return (
        a
        - 4.0
        + (p[25] - 1.0)
        * (
            a
            - 4.0
            + (np.exp(-np.sqrt(p[24] * p[24]) * (a - 4.0)) - 1.0)
            / np.sqrt(p[24] * p[24])
        )
    )


@njit(cache=True)
def _sumex(ex: float) -> float:
    """Eq. A24c."""
    return 1.0 + (1.0 - ex**19.0) / (1.0 - ex) * ex**0.5


@njit(cache=True)
def _sg0(ex: float, p: NDArray[np.float64], ap: NDArray[np.float64]) -> float:
    """Eq. A24a."""
    return (
        _g0(ap[1], p)
        + (
            _g0(ap[2], p) * ex
            + _g0(ap[3], p) * ex * ex
            + _g0(ap[4], p) * ex**3.0
            + (_g0(ap[5], p) * ex**4.0 + _g0(ap[6], p) * ex**12.0)
            * (1.0 - ex**8.0)
            / (1.0 - ex)
        )
    ) / _sumex(ex)


@njit(cache=True)
def msis_gtd7(
    doy: int,
    sec: float,
    alt_km: float,
    lat_deg: float,
    lon_deg: float,
    lst: float,
    f107a: float,
    f107: float,
    ap: float,
    ap_array: NDArray[np.float64],
    effective_density: bool = False,
) -> NRLMSISEOutput:
    switches = np.ones(24)
    if ap_array[0] >= 0.0:
        switches[9] = -1.0
    flags = _make_Flags(switches)
    inp = _make_Input(
        doy, sec, alt_km, lat_deg, lon_deg, lst, f107a, f107, ap, ap_array
    )
    model = _make_Model()
    if effective_density:
        return _modelgtd7d(model, inp, flags)
    return _modelgtd7(model, inp, flags)
