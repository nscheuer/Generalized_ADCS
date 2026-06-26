__all__ = ["thrust_command_to_eci", "ConstantThrust", "ScheduledThrust", "CallableThrust"]

import numpy as np

from ADCS.helpers.math_helpers import rot_mat


def thrust_command_to_eci(accel, frame, q, R, V):
    r"""
    Convert a low-thrust acceleration command to the ECI frame.

    The command is an **acceleration** (the idealized general thrust interface;
    a future ``Thruster`` actuator layer can map force/Isp -> acceleration). It
    may be expressed in any of:

    - ``"ECI"``     : inertial components (returned unchanged),
    - ``"RTN"`` / ``"LVLH"`` : ``[radial, along-track, cross-track]`` in the
      satellite's orbit frame (radial = zenith, along-track ~ velocity, cross
      = orbit normal) -- the natural frame for relative-orbit / station-keeping
      control,
    - ``"BODY"``    : body-frame components, rotated to ECI by the attitude ``q``
      (use this for a physically mounted thruster).

    :param accel: Thrust acceleration components, shape ``(3,)``.
    :param frame: One of ``"ECI"``, ``"RTN"``, ``"LVLH"``, ``"BODY"``.
    :param q: Attitude quaternion (body->ECI), used only for ``"BODY"``.
    :param R: ECI position [km], used only for ``"RTN"``/``"LVLH"``.
    :param V: ECI velocity [km/s], used only for ``"RTN"``/``"LVLH"``.
    :return: Thrust acceleration in ECI, same units as ``accel``.
    """
    accel = np.asarray(accel, dtype=float).reshape(3)
    f = str(frame).upper()
    if f == "ECI":
        return accel.copy()
    if f == "BODY":
        return rot_mat(np.asarray(q, dtype=float)) @ accel
    if f in ("RTN", "LVLH"):
        R = np.asarray(R, dtype=float)
        V = np.asarray(V, dtype=float)
        e_R = R / np.linalg.norm(R)
        e_N = np.cross(R, V)
        e_N = e_N / np.linalg.norm(e_N)
        e_T = np.cross(e_N, e_R)
        return accel[0] * e_R + accel[1] * e_T + accel[2] * e_N
    raise ValueError(f"unknown thrust frame {frame!r} (expected ECI/RTN/LVLH/BODY)")


# --------------------------------------------------------------------------- #
# Pluggable thrust sources.
#
# A "thrust source" is any callable
#     source(t_J2000, x, os, world) -> (accel, frame) | None
# returning a thrust acceleration command [m/s^2] in a named frame, or None for
# coast. ``x`` is the satellite state (quaternion at x[3:7]), ``os`` the current
# orbital state, and ``world`` the FormationWorld (or None) so closed-loop
# formation controllers can read neighbour states. The convenience classes below
# cover the common open-loop cases; pass your own callable for closed-loop /
# adjoint-optimized control.
# --------------------------------------------------------------------------- #
class ConstantThrust:
    r"""Constant open-loop thrust acceleration [m/s^2] in a fixed frame."""

    def __init__(self, accel, frame: str = "RTN") -> None:
        self.accel = np.asarray(accel, dtype=float).reshape(3)
        self.frame = frame

    def __call__(self, t_J2000, x, os, world):
        return (self.accel, self.frame)


class ScheduledThrust:
    r"""
    Piecewise-constant open-loop thrust schedule keyed by J2000 epoch.

    :param times_J2000: Non-decreasing segment start epochs (Julian centuries).
    :param accels: Thrust acceleration [m/s^2] for each segment, shape ``(3,)``.
    :param frame: Frame for all segments (default ``"RTN"``).
    """

    def __init__(self, times_J2000, accels, frame: str = "RTN") -> None:
        self.times = np.asarray(times_J2000, dtype=float)
        self.accels = [np.asarray(a, dtype=float).reshape(3) for a in accels]
        self.frame = frame

    def __call__(self, t_J2000, x, os, world):
        i = int(np.searchsorted(self.times, float(t_J2000), side="right") - 1)
        if i < 0:
            return None
        return (self.accels[min(i, len(self.accels) - 1)], self.frame)


class CallableThrust:
    r"""
    Adapter wrapping a user function ``f(t_J2000, x, os, world) -> accel`` that
    returns a thrust acceleration [m/s^2] in a fixed ``frame`` (or ``None`` to
    coast). Lets a closed-loop control law be supplied without the (accel, frame)
    tuple boilerplate.
    """

    def __init__(self, func, frame: str = "RTN") -> None:
        self.func = func
        self.frame = frame

    def __call__(self, t_J2000, x, os, world):
        a = self.func(t_J2000, x, os, world)
        if a is None:
            return None
        return (np.asarray(a, dtype=float).reshape(3), self.frame)
