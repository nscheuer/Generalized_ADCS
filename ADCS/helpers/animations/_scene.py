"""Scene layers and the current Earth body."""

from pathlib import Path
from typing import Protocol

from ._geometry import uv_sphere

DEFAULT_TEXTURE = Path(__file__).resolve().parents[1] / "plotting/textures/2k_earth_daymap.jpg"


class SceneLayer(Protocol):
    """One independently updated part of the scene."""

    def update(self, time: float) -> None: ...


class SceneGraph:
    """Keep scene composition separate from playback and input handling."""

    def __init__(self) -> None:
        self.layers: dict[str, SceneLayer] = {}

    def add(self, name: str, layer: SceneLayer) -> None:
        if name in self.layers:
            raise ValueError(f"Scene layer already exists: {name}")
        self.layers[name] = layer

    def update(self, time: float) -> None:
        for layer in self.layers.values():
            layer.update(time)


class EarthBody:
    """A textured body whose mesh and material stay on the GPU during playback."""

    def __init__(self, base, radius: float, texture_path=None) -> None:
        from panda3d.core import AmbientLight, DirectionalLight, LColor

        self.root = base.render.attachNewNode("Earth body")
        self.node = self.root.attachNewNode(uv_sphere(radius))
        self.node.setTwoSided(True)

        path = Path(texture_path).expanduser() if texture_path is not None else DEFAULT_TEXTURE
        if not path.is_file():
            raise FileNotFoundError(f"Earth texture not found: {path}")
        texture = base.loader.loadTexture(str(path))
        if texture is None:
            raise RuntimeError(f"Could not load Earth texture: {path}")
        self.node.setTexture(texture, 1)

        ambient = AmbientLight("Earth fill")
        ambient.setColor(LColor(0.24, 0.31, 0.43, 1))
        self.root.setLight(self.root.attachNewNode(ambient))
        sun = DirectionalLight("Earth sunlight")
        sun.setColor(LColor(1.7, 1.62, 1.4, 1))
        sun_node = self.root.attachNewNode(sun)
        sun_node.setHpr(-40, -22, 0)
        self.root.setLight(sun_node)

    def set_orientation(self, quat) -> None:
        self.node.setQuat(quat)


class EarthLayer:
    """Bind Earth's recorded orientation to its persistent render object."""

    def __init__(self, body: EarthBody, orientation_track) -> None:
        self.body = body
        self.orientation_track = orientation_track

    def update(self, time: float) -> None:
        self.body.set_orientation(self.orientation_track.panda_quat(time))
