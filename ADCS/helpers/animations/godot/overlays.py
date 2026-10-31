r"""
Overlay descriptors for the 3-D viewer: vectors, axes, sensor cones, tracks,
target lines and labels drawn over the scene.

Overlays are declared here and stored in the vizlog's ``scene.json``
(``"overlays"``), so a recorded log and a live stream carry the same set. The
viewer only draws what is described; it never decides what to show. Build
them with the classes below or take :func:`default_overlays`.

.. code-block:: python

   from ADCS.helpers.animations.godot import overlays as ov
   ADCS.helpers.animations.godot.results_to_vizlog(results, "run.vizlog", overlays=[
       ov.Arrow("tau_act_body", label="control torque", color="#ff8c1a"),
       ov.Arrow("@sun", label="Sun", scale="fixed_px", px=90, color="#ffe066"),
       ov.SensorCone("startracker0"),
       ov.OrbitTrack(past_s=5400, future_s=1800),
   ])

Vector sources: a channel name (3 values per row; its ``frame`` in the
manifest says BODY or GCRS) or a derived vector: ``@sun`` (spacecraft to Sun),
``@nadir``, ``@velocity``, ``@boresight:<name>`` (a named body boresight).

Arrow lengths: ``physical`` (``units_per_m`` channel units per metre),
``fixed_px`` (constant on screen) or ``log`` (``px_min`` at ``ref``, growing
by ``px_per_decade`` per factor of 10, up to ``px_max``).
"""

__all__ = ["Arrow", "Triad", "SensorCone", "OrbitTrack", "GroundTrack", "TargetLine", "Label",
           "default_overlays", "validate_overlays"]

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

DERIVED = ("@sun", "@nadir", "@velocity")
FRAMES = ("body", "gcrs", "lvlh")
SCALES = ("physical", "fixed_px", "log")


@dataclass
class _Overlay:
    def to_json(self) -> Dict[str, Any]:
        d = {"type": self.TYPE, **{k: v for k, v in asdict(self).items() if v is not None}}
        return d


@dataclass
class Arrow(_Overlay):
    """A vector drawn from the spacecraft's centre of mass."""

    TYPE = "arrow"
    source: str
    label: Optional[str] = None
    color: str = "#ff8c1a"
    scale: str = "log"
    ref: Optional[float] = None          # log: magnitude drawn at px_min (default: 10th percentile of the data)
    px_min: float = 24.0
    px_max: float = 180.0
    px_per_decade: float = 40.0
    px: float = 90.0                     # fixed_px
    units_per_m: Optional[float] = None  # physical
    width_px: float = 3.0
    on_top: bool = True
    visible: bool = True
    id: Optional[str] = None


@dataclass
class Triad(_Overlay):
    """Three axes (X red, Y green, Z blue) at the spacecraft: body, LVLH or GCRS."""

    TYPE = "triad"
    frame: str = "body"
    px: float = 70.0
    on_top: bool = True
    visible: bool = True
    id: Optional[str] = None


@dataclass
class SensorCone(_Overlay):
    """A sensor's field of view (half-angle and boresight from scene.json)."""

    TYPE = "cone"
    sensor: str = ""
    length_px: float = 140.0
    color: str = "#66ccff"
    opacity: float = 0.25
    visible: bool = True
    id: Optional[str] = None


@dataclass
class OrbitTrack(_Overlay):
    """The spacecraft's path in GCRS around the current time."""

    TYPE = "polyline"
    track: str = "orbit"
    past_s: float = 5400.0
    future_s: float = 1800.0
    color: str = "#9fd3ff"
    width_px: float = 2.0
    samples: int = 600
    visible: bool = True
    id: Optional[str] = None


@dataclass
class GroundTrack(_Overlay):
    """The sub-satellite path on the Earth's surface (moves with the Earth)."""

    TYPE = "polyline"
    track: str = "ground"
    past_s: float = 5400.0
    future_s: float = 1800.0
    color: str = "#ffd27f"
    width_px: float = 2.0
    samples: int = 600
    visible: bool = True
    id: Optional[str] = None


@dataclass
class TargetLine(_Overlay):
    """From the spacecraft toward the active pointing target (target_kind/target channels)."""

    TYPE = "target_line"
    color: str = "#7CFC00"
    length_px: float = 220.0   # for direction targets; position targets are drawn to the point
    width_px: float = 2.0
    visible: bool = True
    id: Optional[str] = None


@dataclass
class Label(_Overlay):
    """Text pinned to the spacecraft."""

    TYPE = "label"
    text: str = ""
    offset_px: Sequence[float] = field(default_factory=lambda: (0.0, -40.0))
    color: str = "#ffffff"
    visible: bool = True
    id: Optional[str] = None


def validate_overlays(overlays: Iterable[Any], channels: Dict[str, Any], scene: Dict[str, Any]) -> List[Dict[str, Any]]:
    r"""
    Turn overlays (objects or dicts) into JSON descriptors and check them
    against the log's channels and scene.

    :param channels: ``{name: {"width", "frame", ...}}`` of the log.
    :raises ValueError: On an unknown type, source, frame, scale or sensor.
    """
    out = []
    sensors = {s["id"]: s for s in scene.get("sensors", [])}
    boresights = scene.get("boresights", {})
    for i, o in enumerate(overlays):
        d = o.to_json() if isinstance(o, _Overlay) else dict(o)
        t = d.get("type")
        d.setdefault("id", f"{t}{i}")
        if t == "arrow":
            src = d["source"]
            if src.startswith("@boresight:"):
                if src.split(":", 1)[1] not in boresights:
                    raise ValueError(f"overlay {d['id']}: no boresight {src!r}; have {list(boresights)}")
            elif src.startswith("@"):
                if src not in DERIVED:
                    raise ValueError(f"overlay {d['id']}: unknown derived vector {src!r}; use one of {DERIVED}")
            elif src not in channels or int(channels[src]["width"]) != 3:
                raise ValueError(f"overlay {d['id']}: {src!r} is not a 3-wide channel of this log")
            if d.get("scale") not in SCALES:
                raise ValueError(f"overlay {d['id']}: scale must be one of {SCALES}")
            if d["scale"] == "physical" and not d.get("units_per_m"):
                raise ValueError(f"overlay {d['id']}: physical scale needs units_per_m")
        elif t == "triad":
            if d.get("frame") not in FRAMES:
                raise ValueError(f"overlay {d['id']}: frame must be one of {FRAMES}")
        elif t == "cone":
            s = sensors.get(d.get("sensor"))
            if s is None or "fov_half_rad" not in s:
                raise ValueError(f"overlay {d['id']}: sensor {d.get('sensor')!r} has no field of view in the scene")
        elif t == "polyline":
            if d.get("track") not in ("orbit", "ground"):
                raise ValueError(f"overlay {d['id']}: track must be 'orbit' or 'ground'")
        elif t == "target_line":
            if "target" not in channels or "target_kind" not in channels:
                raise ValueError(f"overlay {d['id']}: the log has no target channels")
        elif t != "label":
            raise ValueError(f"unknown overlay type {t!r}")
        out.append(d)
    return out


def default_overlays(channels: Dict[str, Any], scene: Dict[str, Any], ref: Optional[Dict[str, float]] = None) -> List[_Overlay]:
    r"""
    A sensible set for what the log contains: body axes, control and
    disturbance torques, wheel momentum, Sun direction, magnetic field, sensor
    cones, orbit and ground tracks, and the pointing target.

    :param ref: Optional ``{channel: magnitude}`` for log-scaled arrows (e.g.
        the 10th percentile of each channel's norm), so small values stay visible.
    """
    ref = ref or {}
    out: List[_Overlay] = [Triad("body"), Triad("gcrs", visible=False), Triad("lvlh", visible=False),
                           Arrow("@nadir", label="Nadir", color="#6ee7ff", scale="fixed_px", visible=False),
                           Arrow("@velocity", label="Along track", color="#82ef94", scale="fixed_px", visible=False)]
    for axis, color in enumerate(("#ff6666", "#66ff66", "#6666ff")):
        out.append(Arrow(f"user_inertia_axis_{axis}", label=f"Principal axis {axis + 1}",
                         color=color, scale="fixed_px", visible=False))
    for name in scene.get("boresights", {}):
        out.append(Arrow(f"@boresight:{name}", label=f"Boresight {name}", color="#ffdd66",
                         scale="fixed_px", visible=False))
    vectors = [("tau_act_body", "control torque", "#ff8c1a"), ("tau_dist_body", "disturbance torque", "#ff4d4d"),
               ("h_rw_body", "wheel momentum", "#33e0c8"), ("b_field", "magnetic field", "#d47cff")]
    for ch, label, color in vectors:
        if ch in channels:
            if ch == "b_field":
                out.append(Arrow(ch, label=label, color=color, scale="fixed_px", px=80.0))
            else:
                out.append(Arrow(ch, label=label, color=color, scale="log", ref=ref.get(ch)))
    out.append(Arrow("@sun", label="Sun", color="#ffe066", scale="fixed_px", px=100.0))
    for s in scene.get("sensors", []):
        if "fov_half_rad" in s and not s.get("fov_assumed", False):
            out.append(SensorCone(s["id"]))
    out += [OrbitTrack(), GroundTrack()]
    if "target" in channels and "target_kind" in channels:
        out.append(TargetLine())
    return out
