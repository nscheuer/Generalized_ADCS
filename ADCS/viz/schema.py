r"""
Channel registry for the ``adcs-vizlog`` format, version 1.

The normative description lives in ``docs/source/viz/vizlog_v1.rst``. This
module only depends on the standard library so that the format definition can
be read, and ported to other languages, without the rest of ADCS.
"""

from __future__ import annotations

__all__ = [
    "FORMAT",
    "SCHEMA_VERSION",
    "QUATERNION_CONVENTION",
    "Channel",
    "REQUIRED_CHANNELS",
    "OPTIONAL_CHANNELS",
    "TARGET_KIND",
    "channel_spec",
]

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional

FORMAT = "adcs-vizlog"
SCHEMA_VERSION = 1
QUATERNION_CONVENTION = (
    "Hamilton, scalar-first [q0, q1, q2, q3], active; q_A_B rotates vectors "
    "expressed in frame B into frame A (v_A = R(q_A_B) v_B)"
)

#: Encoding of the ``target_kind`` channel.
TARGET_KIND = {"none": 0, "direction": 1, "quaternion": 2, "position": 3}


@dataclass(frozen=True)
class Channel:
    r"""
    One time-series column of a vizlog run.

    Stored as little-endian float64, row-major, shape ``(n_rows, width)``.

    :param name: Channel name; also its file stem (``<name>.f64``).
    :param width: Values per row.
    :param unit: Physical unit (``"-"`` if dimensionless).
    :param frame: Frame the values are expressed in, or ``"-"``.
    :param required: Whether every vizlog must contain it.
    :param description: Human-readable meaning.
    :param labels: Optional per-column labels (length ``width``).
    :param provenance: Optional note on how the values were obtained.
    """

    name: str
    width: int
    unit: str
    frame: str
    required: bool = False
    description: str = ""
    labels: Optional[List[str]] = field(default=None)
    provenance: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> Channel:
        return cls(**data)


def _ch(*args: Any, **kwargs: Any) -> Channel:
    return Channel(*args, **kwargs)


#: Channels every vizlog run contains. ``width`` is fixed.
REQUIRED_CHANNELS: Dict[str, Channel] = {c.name: c for c in [
    _ch("t_s", 1, "s", "-", True, "TT seconds since manifest epoch"),
    _ch("r_sc", 3, "km", "GCRS", True, "spacecraft position"),
    _ch("v_sc", 3, "km/s", "GCRS", True, "spacecraft velocity"),
    _ch("q_gcrs_body", 4, "-", "-", True,
        "attitude; sign chosen for continuity with the previous row"),
    _ch("w_body", 3, "rad/s", "BODY", True, "angular velocity of BODY relative to GCRS"),
    _ch("r_sun", 3, "km", "GCRS", True, "apparent geocentric Sun position"),
    _ch("q_itrs_gcrs", 4, "-", "-", True, "Earth orientation (GCRS to ITRS)"),
    _ch("shadow_factor", 1, "-", "-", True,
        "visible fraction of the solar disk (conical model, 0..1)"),
]}

#: Optional channels. ``width`` of ``0`` means it is set per log (e.g. number of wheels).
OPTIONAL_CHANNELS: Dict[str, Channel] = {c.name: c for c in [
    _ch("r_moon", 3, "km", "GCRS", description="apparent geocentric Moon position"),
    _ch("sunlit_binary", 1, "-", "-",
        description="eclipse flag the simulation used (1 sunlit, 0 eclipse)"),
    _ch("b_field", 3, "T", "GCRS", description="geomagnetic field at the spacecraft"),
    _ch("rho", 1, "kg/m^3", "-", description="atmospheric density"),
    _ch("h_wheel", 0, "N m s", "-", description="wheel momentum relative to the body, per wheel"),
    _ch("wheel_speed", 0, "rad/s", "-", description="wheel spin rate relative to the body, per wheel"),
    _ch("u_cmd", 0, "mixed", "-",
        description="actuator commands in satellite.actuators order (MTQ A m^2, RW N m)"),
    _ch("tau_act_body", 3, "N m", "BODY", description="total actuator torque"),
    _ch("tau_dist_body", 3, "N m", "BODY", description="total disturbance torque"),
    _ch("target_kind", 1, "-", "-", description="goal encoding, see TARGET_KIND"),
    _ch("target", 4, "mixed", "GCRS",
        description="kind 1: [nan, unit dir]; 2: q_gcrs_body target; 3: [nan, position km]"),
    _ch("w_target", 3, "rad/s", "GCRS", description="target angular velocity"),
    _ch("boresight_body", 3, "-", "BODY", description="boresight of the active goal"),
    _ch("pointing_err", 1, "rad", "-", description="angle between attitude and active goal"),
    _ch("q_est", 4, "-", "-", description="estimated q_gcrs_body"),
    _ch("w_est", 3, "rad/s", "BODY", description="estimated angular velocity"),
    _ch("r_est", 3, "km", "GCRS", description="estimated spacecraft position"),
    _ch("v_est", 3, "km/s", "GCRS", description="estimated spacecraft velocity"),
]}


def channel_spec(name: str, width: Optional[int] = None, **overrides: Any) -> Channel:
    r"""
    Look up a registered channel, filling in a per-log width if needed.

    Unregistered names starting with ``sensor_`` are allowed (one channel per
    attitude sensor); they must be given ``width`` and ``unit``.

    :param name: Channel name.
    :param width: Width for variable-width channels (registry width ``0``).
    :param overrides: Fields to replace, e.g. ``labels`` or ``provenance``.
    :return: The channel description.
    :raises KeyError: If the name is neither registered nor a sensor channel.
    :raises ValueError: If a variable-width channel is given no width, or a
        fixed-width channel a different one.
    """
    spec = REQUIRED_CHANNELS.get(name) or OPTIONAL_CHANNELS.get(name)
    if spec is None:
        if not name.startswith("sensor_"):
            raise KeyError(f"unknown vizlog channel {name!r}")
        if width is None:
            raise ValueError(f"sensor channel {name!r} needs a width")
        spec = Channel(name, int(width), overrides.pop("unit", "-"), overrides.pop("frame", "-"))
    elif spec.width == 0:
        if width is None or int(width) < 1:
            raise ValueError(f"channel {name!r} is variable-width; pass width >= 1")
        spec = replace(spec, width=int(width))
    elif width is not None and int(width) != spec.width:
        raise ValueError(f"channel {name!r} has width {spec.width}, got {width}")
    return replace(spec, **overrides) if overrides else spec
