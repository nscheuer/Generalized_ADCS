"""Scene layers and the current Earth body."""

from pathlib import Path
from typing import Protocol

from ._geometry import uv_sphere
from ._earth_material import EarthMaterial
from ._texture_assets import texture_path as asset_texture_path, tier_name

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

    def __init__(self, base, radius: float, texture_path=None, tier="medium") -> None:
        from panda3d.core import AmbientLight, DirectionalLight, LColor, Shader
        from ._earth_material import VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER

        self.root = base.render.attachNewNode("Earth body")
        self.node = self.root.attachNewNode(uv_sphere(radius))
        self.node.setTwoSided(True)
        self.node.setDepthTest(True)
        self.node.setDepthWrite(True)

        self._custom_texture = Path(texture_path).expanduser() if texture_path is not None else None
        path = self._custom_texture or asset_texture_path(tier_name(tier, '2k_earth_daymap.jpg'))
        if not path.is_file():
            raise FileNotFoundError(f"Earth texture not found: {path}")
        self._base = base
        self.texture_path = path
        self._material = EarthMaterial(base, self.node, self._custom_texture, tier)
        self._simple_shader = Shader.make(Shader.SL_GLSL, VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER)
        self._texture_enabled = False
        self.lod = 'color'
        self._low_graphics = tier == 'low'
        self.graphics_quality = tier
        self._legacy_texture = None
        self._sun_direction_root = (0.0, 0.0, 1.0)
        self.node.setColor(0.33, 0.55, 0.83, 1)

        ambient = AmbientLight("Earth fill")
        # Keep the night side near-black; most visible illumination comes
        # from the directional sunlight below.
        ambient.setColor(LColor(0.012, 0.016, 0.025, 1))
        self.root.setLight(self.root.attachNewNode(ambient))
        sun = DirectionalLight("Earth sunlight")
        sun.setColor(LColor(1.7, 1.62, 1.4, 1))
        sun_node = self.root.attachNewNode(sun)
        self.root.setLight(sun_node)
        self.sunlight = sun_node

    def set_sun_direction(self, vector) -> None:
        """Update the Sun vector in the rotating Earth's local frame."""
        from panda3d.core import Point3
        self._sun_direction_root = tuple(float(v) for v in vector)
        # Panda's directional light emits along its local +Y axis, so aim its
        # node away from the incoming Sun vector.
        self.sunlight.lookAt(self.root, Point3(*(-v for v in self._sun_direction_root)))
        direction = self.node.getQuat(self.root).conjugate().xform(Point3(*self._sun_direction_root))
        self._material.set_sun_direction(direction)
        if self.lod == 'simple_2k':
            self.node.setShaderInput('sun_direction', *direction)

    def set_orientation(self, quat) -> None:
        self.node.setQuat(quat)
        self.set_sun_direction(self._sun_direction_root)

    def update_camera(self) -> None:
        """Refresh the view vector used for the ocean specular highlight."""
        from panda3d.core import Point3
        camera_position = self.node.getRelativePoint(self._base.camera, Point3(0, 0, 0))
        self._material.set_camera_position(camera_position)

    def set_cloud_time(self, utc: float) -> None:
        # A five-day drift cycle gives the cloud deck visible motion while
        # remaining tied to playback and independent of render frame rate.
        self._material.set_cloud_offset(utc / (86400.0 * 5.0))

    def set_texture_enabled(self, enabled: bool) -> None:
        self.set_lod('simple_2k' if enabled and self.graphics_quality == 'low'
                     else 'advanced_2k' if enabled else 'color')

    def set_lod(self, lod: str) -> None:
        """Swap Earth's material only when its screen-space detail stage changes."""
        if lod == self.lod:
            return
        if self._texture_enabled:
            self._release_texture_quality()
        self.lod = lod
        self._texture_enabled = lod != 'color'
        if self._texture_enabled:
            self.node.setColor(1, 1, 1, 1)
            self._apply_texture_quality()
        else:
            self.node.setColor(0.33, 0.55, 0.83, 1)

    @staticmethod
    def _fade(value: float, start: float, end: float) -> float:
        fraction = max(0.0, min(1.0, (value - start) / (end - start)))
        return fraction * fraction * (3.0 - 2.0 * fraction)

    def set_lod_detail(self, diameter_pixels: float) -> None:
        """Fade in Earth detail as its visible disc grows on screen."""
        texture_blend = self._fade(diameter_pixels, 6.0, 32.0)
        if self.lod == 'simple_2k':
            self.node.setShaderInput('texture_blend', texture_blend)
        elif self.lod.startswith('advanced_'):
            detail_strength = self._fade(diameter_pixels, 48.0, 160.0)
            resolution_blend = (self._fade(diameter_pixels, 160.0, 400.0)
                                if self.lod == 'advanced_8k' else 1.0)
            self._material.set_lod_blend(texture_blend, detail_strength,
                                         resolution_blend)
            normal_strength = (0.48 * self._fade(diameter_pixels, 160.0, 520.0)
                               if self.lod == 'advanced_8k' else 0.0)
            self._material.set_normal_strength(normal_strength)

    @property
    def low_graphics(self) -> bool:
        return self._low_graphics

    def set_low_graphics(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled == self._low_graphics:
            return
        self._low_graphics = enabled
        if self._texture_enabled:
            self._release_texture_quality()
            self._apply_texture_quality()

    def set_graphics_quality(self, tier: str) -> None:
        if tier == self.graphics_quality:
            return
        if self._texture_enabled:
            self._release_texture_quality()
        self._texture_enabled = False
        self.lod = 'color'
        self.node.setColor(0.33, 0.55, 0.83, 1)
        self.graphics_quality = tier
        self._low_graphics = tier == 'low'
        self.texture_path = self._custom_texture or asset_texture_path(tier_name(tier, '2k_earth_daymap.jpg'))
        self._material.set_quality(tier)

    def _apply_texture_quality(self) -> None:
        if self.lod == 'simple_2k':
            self._material.set_enabled(False)
            self.texture_path = self._custom_texture or asset_texture_path('2k_earth_daymap.jpg')
            self._legacy_texture = self._base.loader.loadTexture(str(self.texture_path))
            if self._legacy_texture is None:
                raise RuntimeError(f"Could not load Earth texture: {self.texture_path}")
            self.node.setShader(self._simple_shader, 1)
            self.node.setShaderInput('day_map', self._legacy_texture)
            self.node.setShaderInput('sun_direction', *self._material.sun_direction)
            self.node.setShaderInput('texture_blend', 1.0)
        else:
            self._material.set_quality('high' if self.lod == 'advanced_8k' else 'medium')
            self._material.set_normal_strength(0.48 if self.lod == 'advanced_8k' else 0.0)
            self._material.set_enabled(True)

    def _release_texture_quality(self) -> None:
        self._material.set_enabled(False)
        if self.lod == 'simple_2k':
            for name in ('day_map', 'sun_direction', 'texture_blend'):
                self.node.clearShaderInput(name)
            self.node.clearShader()
        self.node.clearTexture()
        if self._legacy_texture is not None:
            self._base.loader.unloadTexture(self._legacy_texture)
            self._legacy_texture = None

    def close(self) -> None:
        self._release_texture_quality()
        self._material.close()


class EarthLayer:
    """Bind Earth's recorded orientation to its persistent render object."""

    def __init__(self, body: EarthBody, orientation_track) -> None:
        self.body = body
        self.orientation_track = orientation_track

    def update(self, time: float) -> None:
        self.body.set_orientation(self.orientation_track.panda_quat(time))
