"""Physical-scale solar-system bodies with point, mesh, and texture LOD."""

from dataclasses import dataclass
from datetime import datetime, timezone
from math import radians, sqrt, tan
from pathlib import Path

import numpy as np
from skyfield.errors import EphemerisRangeError

from ._ephemeris_track import PlanetaryEphemerisTrack
from ._body_rotation import body_orientation_quaternion
from ._geometry import uv_sphere
from ._orbit_trails import OrbitTrail
from ._saturn_rings import SaturnRings
from ._texture_assets import texture_path, tier_name


TEXTURE_DIR = Path(__file__).resolve().parents[1] / 'plotting' / 'textures'


@dataclass(frozen=True)
class BodySpec:
    name: str
    radius_km: float
    color: tuple[float, float, float, float]
    texture_name: str | None = None


# Radii are half the planetary diameters in NASA's Planetary Fact Sheet.
# Sun radius is the NASA nominal solar radius; Earth uses the ADCS constant.
BODY_SPECS = (
    BodySpec('sun', 695700.0, (1.00, 0.91, 0.68, 1), '2k_sun.jpg'),
    BodySpec('mercury', 2439.5, (0.73, 0.69, 0.65, 1), '2k_mercury.jpg'),
    BodySpec('venus', 6052.0, (0.92, 0.78, 0.58, 1), '2k_venus_atmosphere.jpg'),
    BodySpec('earth', 6378.1363, (0.41, 0.69, 1.00, 1)),
    BodySpec('moon', 1737.5, (0.75, 0.75, 0.73, 1), '2k_moon.jpg'),
    BodySpec('mars', 3396.0, (0.94, 0.45, 0.32, 1), '2k_mars.jpg'),
    BodySpec('jupiter', 71492.0, (0.92, 0.74, 0.57, 1), '2k_jupiter.jpg'),
    BodySpec('saturn', 60268.0, (0.91, 0.81, 0.58, 1), '2k_saturn.jpg'),
    BodySpec('uranus', 25559.0, (0.62, 0.84, 0.91, 1), '2k_uranus.jpg'),
    BodySpec('neptune', 24764.0, (0.39, 0.57, 0.99, 1), '2k_neptune.jpg'),
)


class CelestialBody:
    def __init__(self, base, spec: BodySpec, sphere_geometry, earth_body=None, tier='medium') -> None:
        self.base = base
        self.spec = spec
        self.earth_body = earth_body
        self.texture = None
        self.lod = 'color'
        self.tier = tier
        self.texture_path = texture_path(tier_name(tier, spec.texture_name)) if spec.texture_name else None
        self.texture_available = self.texture_path is not None
        if earth_body is not None:
            self.root = earth_body.root
            self.mesh = earth_body.node
        else:
            self.root = base.render.attachNewNode(spec.name)
            # Panda nodes own render state such as transform and draw mask.
            # Give each body a shallow GeomNode copy so LOD visibility and
            # physical scale stay independent while the immutable Geom data
            # remains shared between planets.
            self.mesh = self.root.attachNewNode(sphere_geometry.makeCopy())
            self.mesh.setScale(spec.radius_km)
            self.mesh.setTwoSided(True)
            self.mesh.setDepthTest(True)
            self.mesh.setDepthWrite(True)
            self.mesh.setColor(*spec.color)
            if spec.name == 'sun':
                self.mesh.setLightOff()
                self.mesh.setColorScale(3.0, 2.4, 1.6, 1.0)
            else:
                from panda3d.core import AmbientLight, DirectionalLight, LColor
                ambient = AmbientLight(f'{spec.name} fill')
                ambient.setColor(LColor(0.008, 0.010, 0.016, 1))
                self.root.setLight(self.root.attachNewNode(ambient))
                sun = DirectionalLight(f'{spec.name} sunlight')
                sun.setColor(LColor(1.6, 1.52, 1.36, 1))
                self.sunlight = self.root.attachNewNode(sun)
                self.root.setLight(self.sunlight)

    def set_texture_enabled(self, enabled: bool) -> None:
        self.set_lod('2k' if enabled else 'color')

    def set_lod(self, lod: str) -> None:
        if self.earth_body is not None:
            self.earth_body.set_lod(lod)
            return
        if lod == self.lod or not self.texture_available:
            return
        if self.texture is not None:
            self.mesh.clearTexture()
            self.base.loader.unloadTexture(self.texture)
            self.texture = None
        if lod != 'color':
            name = tier_name('high' if lod == '8k' else 'medium', self.spec.texture_name)
            self.texture_path = texture_path(name)
            texture = self.base.loader.loadTexture(str(self.texture_path))
            if texture is None:
                raise RuntimeError(f'Could not load texture: {self.texture_path}')
            texture.setMinfilter(texture.FTLinearMipmapLinear)
            texture.setMagfilter(texture.FTLinear)
            self.texture = texture
            self.mesh.setColor(1, 1, 1, 1)
            self.mesh.setTexture(texture, 1)
        else:
            self.mesh.setColor(*self.spec.color)
        self.lod = lod

    def set_quality(self, tier):
        self.tier = tier

    def set_sun_direction(self, vector) -> None:
        if self.earth_body is not None:
            self.earth_body.set_sun_direction(vector)
            return
        if hasattr(self, 'sunlight'):
            from panda3d.core import Point3
            # Directional lights emit along local +Y; point the light node
            # opposite the vector from this body toward the Sun.
            self.sunlight.lookAt(self.root, Point3(*(-float(v) for v in vector)))


class SolarSystemLayer:
    """Place bodies at ephemeris positions and swap detail by angular size."""

    FOCUS_ORDER = tuple(spec.name for spec in BODY_SPECS)
    LABEL_MIN_DISTANCE_KM = 3.0e7

    @staticmethod
    def _choose_lod(current, stages, diameter, enter, leave):
        """Use different entry and exit sizes to prevent zoom-boundary flicker."""
        index = stages.index(current) if current in stages else 0
        while index + 1 < len(stages) and diameter >= enter[index]:
            index += 1
        while index > 0 and diameter < leave[index - 1]:
            index -= 1
        return stages[index]

    def set_quality(self, tier):
        for body in self.bodies.values():
            if body.earth_body is None:
                body.set_quality(tier)
        self.saturn_rings.set_quality(tier)

    def __init__(self, base, earth_body, ephemeris=None, tier='medium') -> None:
        self.base = base
        self.ephemeris_track = PlanetaryEphemerisTrack(ephemeris)
        sphere_geometry = uv_sphere(1.0)
        self.bodies = {
            spec.name: CelestialBody(base, spec, sphere_geometry,
                                     earth_body if spec.name == 'earth' else None, tier)
            for spec in BODY_SPECS
        }
        saturn = self.bodies['saturn']
        self.saturn_rings = SaturnRings(base, saturn.root, saturn.spec.radius_km, tier)
        self.positions = {'earth': np.zeros(3, dtype=np.float64)}
        self.focus_name = 'earth'
        self._sampled_utc = None
        self._rotation_utc = None
        self.available = False
        self._tags = {}
        self._markers = {}
        self._label_font = None
        self._orbit_trails = {
            name: OrbitTrail(base.render, body.spec.color)
            for name, body in self.bodies.items() if name != 'sun'
        }
        for name, body in self.bodies.items():
            self._markers[name] = self._make_marker(name, body.spec.color)

    def _make_marker(self, name, color):
        from panda3d.core import CardMaker, Shader, TransparencyAttrib
        maker = CardMaker(f'{name} marker')
        maker.setFrame(-1, 1, -1, 1)
        node = self.base.aspect2d.attachNewNode(maker.generate())
        node.setColor(*(1.0, 0.92, 0.70, 1) if name == 'sun' else color)
        node.setTwoSided(True)
        node.setDepthTest(False)
        node.setDepthWrite(False)
        node.setBin('fixed', 1)
        if name == 'sun':
            node.setTransparency(TransparencyAttrib.MAlpha)
            masks = '\n'.join(
                f'    if (occluder{i}.w > 0.0) alpha *= smoothstep('
                f'occluder{i}.z - 0.002, occluder{i}.z + 0.002, '
                f'length(uv - occluder{i}.xy));'
                for i in range(len(BODY_SPECS) - 1)
            )
            uniforms = '\n'.join(
                f'uniform vec4 occluder{i};' for i in range(len(BODY_SPECS) - 1)
            )
            node.setShader(Shader.make(Shader.SL_GLSL, '''#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
varying vec2 uv;
void main() { uv = p3d_MultiTexCoord0; gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex; }
''', f'''#version 120
varying vec2 uv;
{uniforms}
void main() {{
    float r = length(uv * 2.0 - 1.0);
    float core = 1.0 - smoothstep(0.08, 0.18, r);
    float halo = pow(max(0.0, 1.0 - r), 1.5);
    float alpha = max(core, halo * 0.85);
{masks}
    gl_FragColor = vec4(mix(vec3(1.0, 0.55, 0.08), vec3(1.0), core),
                        alpha);
}}
'''))
        node.hide()
        return node

    def set_label_font(self, font) -> None:
        from direct.gui.OnscreenText import OnscreenText
        from panda3d.core import TextNode
        self._label_font = font
        for name, body in self.bodies.items():
            label = OnscreenText(text=name.upper(), pos=(0, 0), scale=0.027,
                                 align=TextNode.ALeft, fg=body.spec.color,
                                 font=font, mayChange=True,
                                 parent=self.base.aspect2d)
            label.setShadow((0, 0, 0, 0.8))
            label.hide()
            self._tags[name] = label
        self.resize_ui()

    def resize_ui(self) -> None:
        """Preserve label height as aspect2d adjusts for tall windows."""
        scale = 0.027 * max(1.0, self.base.a2dTop)
        for label in self._tags.values():
            label.setTextScale(scale)

    def update(self, utc: float, playback_speed: float = 1.0) -> None:
        self.update_rotations(utc)
        # The Moon moves fastest. A one-second minimum avoids kernel work on
        # every real-time frame; accelerated playback samples as needed.
        interval = max(1.0, min(60.0, abs(playback_speed) * 0.1))
        if self._sampled_utc is not None and abs(utc - self._sampled_utc) < interval:
            return
        try:
            self.positions = self.ephemeris_track.positions_at_utc(utc)
        except EphemerisRangeError:
            self.available = False
            for name, body in self.bodies.items():
                if name != 'earth':
                    body.root.hide()
            return
        self.available = True
        self._sampled_utc = utc
        for body in self.bodies.values():
            body.root.show()
        self._apply_positions()

    def update_rotations(self, utc: float) -> None:
        """Set each body's IAU orientation at the viewer's current UTC time."""
        utc = float(utc)
        self.bodies['earth'].earth_body.set_cloud_time(utc)
        if self._rotation_utc == utc:
            return
        epoch = self.ephemeris_track.ephemeris.ts.from_datetime(
            datetime.fromtimestamp(utc, timezone.utc)
        )
        tdb_julian_date = float(epoch.tdb)
        for name, body in self.bodies.items():
            orientation = body_orientation_quaternion(name, tdb_julian_date)
            body.mesh.setQuat(orientation)
            if name == 'saturn':
                self.saturn_rings.set_orientation(orientation)
        self._rotation_utc = utc

    def _apply_positions(self) -> None:
        origin = self.positions[self.focus_name]
        sun = self.positions['sun']
        for name, body in self.bodies.items():
            position = self.positions[name]
            body.root.setPos(*(position - origin))
            if name != 'sun':
                body.set_sun_direction(sun - position)

    def set_focus(self, name: str) -> CelestialBody:
        name = name.lower()
        if name not in self.bodies:
            raise ValueError(f'Unknown solar-system body: {name}')
        self.focus_name = name
        if self.available:
            self._apply_positions()
        return self.bodies[name]

    def pick_body(self, mouse):
        """Return the nearest visible body under a pointer in scene space."""
        if mouse is None:
            return None
        from panda3d.core import LPoint2f

        region = self.base.camNode.getDisplayRegion(0)
        height = region.getPixelHeight()
        width = self.base.win.getXSize() if self.base.win is not None else 0
        if height <= 0 or width <= 0:
            return None
        focal_pixels = height / (2.0 * tan(radians(self.base.camLens.getVfov()) / 2.0))
        pixels_per_unit_x = width / (self.base.a2dRight - self.base.a2dLeft)
        pixels_per_unit_y = height / (self.base.a2dTop - self.base.a2dBottom)
        aspect = self.base.a2dRight
        pointer_x = float(mouse.x) * aspect
        pointer_y = float(mouse.y) * self.base.a2dTop
        viewport_bottom = region.getBottom()
        camera_position = self.base.camera.getPos(self.base.render)
        hits = []

        for name, body in self.bodies.items():
            if name != 'earth' and not self.available:
                continue
            world_position = body.root.getPos(self.base.render)
            camera_point = self.base.camera.getRelativePoint(self.base.render,
                                                             world_position)
            depth = float(camera_point.y)
            if depth <= 0:
                continue
            projected = LPoint2f()
            if not self.base.camLens.project(camera_point, projected):
                continue
            px, py = float(projected.x), float(projected.y)
            if abs(px) > 1.1 or abs(py) > 1.1:
                continue
            center_x = px * aspect
            center_y = (viewport_bottom + (1.0 - viewport_bottom) * py) * self.base.a2dTop
            distance = max(1.0, (camera_position - world_position).length())
            radius_pixels = body.spec.radius_km * focal_pixels / distance
            if radius_pixels < 1.5:
                radius_pixels = 2.0 if name == 'sun' else 1.25
            # A modest pointer target makes distant planet markers easy to
            # select without making nearby bodies feel imprecise.
            hit_radius = max(8.0, radius_pixels + 3.0)
            dx = (pointer_x - center_x) * pixels_per_unit_x
            dy = (pointer_y - center_y) * pixels_per_unit_y
            if dx * dx + dy * dy <= hit_radius * hit_radius:
                hits.append((depth, name))

        return min(hits)[1] if hits else None

    def update_lod(self) -> None:
        region = self.base.camNode.getDisplayRegion(0)
        height = region.getPixelHeight()
        if height <= 0:
            return
        focal_pixels = height / (2.0 * tan(radians(self.base.camLens.getVfov()) / 2.0))
        camera_position = self.base.camera.getPos(self.base.render)
        labels = []
        viewport_bottom = self.base.camNode.getDisplayRegion(0).getBottom()
        aspect = self.base.a2dRight
        window_width = self.base.win.getXSize() if self.base.win is not None else 0
        pixels_per_unit = (window_width / (self.base.a2dRight - self.base.a2dLeft)
                           if window_width > 0 else height / 2.0)
        marker_projections = {}
        sun_marker_pixels = None
        for name, body in self.bodies.items():
            self._markers[name].hide()
            tag = self._tags.get(name)
            if tag is not None:
                tag.hide()
            if not self.available and name != 'earth':
                body.set_texture_enabled(False)
                if name == 'saturn':
                    self.saturn_rings.set_visible(False)
                continue
            world_position = body.root.getPos(self.base.render)
            relative_position = camera_position - world_position
            distance = max(1.0, relative_position.length())
            diameter_pixels = 2.0 * body.spec.radius_km * focal_pixels / distance
            if name == 'saturn':
                self.saturn_rings.set_visible(diameter_pixels >= 1.5)
            if diameter_pixels >= 3.0:
                body.mesh.show()
            else:
                body.mesh.hide()
            if name == 'earth':
                stages = ['color', ('simple_2k' if body.earth_body.graphics_quality == 'low'
                                    else 'advanced_2k')]
                enter, leave = [6.0], [4.0]
                if body.earth_body.graphics_quality == 'high':
                    stages.append('advanced_8k')
                    enter.append(160.0)
                    leave.append(120.0)
                current = body.earth_body.lod
            else:
                stages = ['color', '2k']
                enter, leave = [32.0], [20.0]
                if body.tier == 'high' and tier_name('high', body.spec.texture_name) != body.spec.texture_name:
                    stages.append('8k')
                    enter.append(640.0)
                    leave.append(480.0)
                current = body.lod
            body.set_lod(self._choose_lod(current, stages, diameter_pixels, enter, leave))
            if name == 'earth':
                body.earth_body.set_lod_detail(diameter_pixels)

            camera_point = self.base.camera.getRelativePoint(self.base.render,
                                                             world_position)
            from panda3d.core import LPoint2f
            projected = LPoint2f()
            projection_ok = self.base.camLens.project(camera_point, projected)
            if projection_ok:
                px, py = float(projected.x), float(projected.y)
            else:
                # Keep a direction for targets behind the camera so they can
                # still receive an edge tag.
                depth = max(abs(float(camera_point.y)), 1e-8)
                h_fov, v_fov = self.base.camLens.getHfov(), self.base.camLens.getVfov()
                px = float(camera_point.x) / depth / tan(radians(h_fov) / 2)
                py = float(camera_point.z) / depth / tan(radians(v_fov) / 2)
                if camera_point.y < 0:
                    px, py = -px, -py
                    if abs(px) + abs(py) < 0.05:
                        py = 1.0
            in_front = camera_point.y > 0
            inside = projection_ok and in_front and abs(px) <= 1 and abs(py) <= 1
            if in_front:
                marker_projections[name] = (px, py, distance)
            if inside:
                marker_x = px * aspect
                marker_y = (viewport_bottom + (1 - viewport_bottom) * py) * self.base.a2dTop
                if name == 'sun' or diameter_pixels < 3.0:
                    marker = self._markers[name]
                    if name == 'sun':
                        radius = body.spec.radius_km
                        # The tangent to a nearby sphere grows much faster
                        # than the distant-object estimate used by planet LOD.
                        tangent_distance = sqrt(max(distance * distance - radius * radius, 1.0))
                        visible_diameter = 2.0 * focal_pixels * radius / tangent_distance
                        marker_pixels = max(64.0, visible_diameter * 4.0)
                        sun_marker_pixels = marker_pixels
                    else:
                        marker_pixels = 2.5
                    marker.setScale(marker_pixels / (2.0 * pixels_per_unit))
                    marker.setPos(marker_x, 0, marker_y)
                    marker.show()
                if distance >= self.LABEL_MIN_DISTANCE_KM and tag is not None:
                    body_half_pixels = max(1.25, diameter_pixels / 2.0)
                    label_x = marker_x + (body_half_pixels + 0.5) / pixels_per_unit
                    labels.append((name, label_x, marker_y - 0.006,
                                   f'< {name.upper()}'))

        # The HUD halo is outside the scene depth buffer. Mask each foreground
        # planet's projected disc so the Sun cannot shine through its sphere.
        sun_projection = marker_projections.get('sun')
        if sun_projection is not None and sun_marker_pixels is not None:
            sun_x, sun_y, sun_distance = sun_projection
            sun_marker = self._markers['sun']
            occluder_index = 0
            for name, body in self.bodies.items():
                if name == 'sun':
                    continue
                projection = marker_projections.get(name)
                mask = (0.0, 0.0, 0.0, 0.0)
                if projection is not None:
                    px, py, distance = projection
                    radius = body.spec.radius_km
                    if distance - radius < sun_distance and distance - radius < self.base.camLens.getFar():
                        radius_pixels = focal_pixels * radius / sqrt(max(distance * distance - radius * radius, 1.0))
                        u = 0.5 + (px - sun_x) * window_width / (2.0 * sun_marker_pixels)
                        v = 0.5 + (py - sun_y) * height / (2.0 * sun_marker_pixels)
                        mask = (u, v, radius_pixels / sun_marker_pixels, 1.0)
                sun_marker.setShaderInput(f'occluder{occluder_index}', *mask)
                occluder_index += 1

        # Small planet markers are also HUD cards; hide a marker when its
        # sightline passes through a nearer, rendered planetary sphere.
        for name, body in self.bodies.items():
            projection = marker_projections.get(name)
            if projection is None:
                continue
            px, py, distance = projection
            radius = body.spec.radius_km
            if distance - radius >= self.base.camLens.getFar():
                continue
            radius_pixels = focal_pixels * radius / sqrt(max(distance * distance - radius * radius, 1.0))
            for target, target_projection in marker_projections.items():
                if target == name or target == 'sun':
                    continue
                tx, ty, target_distance = target_projection
                dx = (tx - px) * window_width / 2.0
                dy = (ty - py) * height / 2.0
                if distance < target_distance and dx * dx + dy * dy < radius_pixels * radius_pixels:
                    self._markers[target].hide()

        visible_label_names = {name for name, *_ in labels}
        if labels:
            for name, x, y, text in labels:
                label = self._tags[name]
                label.setText(text)
                from panda3d.core import TextNode
                label.setAlign(TextNode.ALeft)
                label.setPos(x, y)
                label.show()

        for name, trail in self._orbit_trails.items():
            if (self.available and name in visible_label_names
                    and self._sampled_utc is not None):
                trail.update(self.ephemeris_track, name, self._sampled_utc,
                             self.positions, self.focus_name)
            else:
                trail.hide()

    def close(self) -> None:
        for trail in self._orbit_trails.values():
            if trail.node is not None:
                trail.node.removeNode()
        self.saturn_rings.close()
        for body in self.bodies.values():
            body.set_texture_enabled(False)
        for marker in self._markers.values():
            marker.removeNode()
        for tag in self._tags.values():
            tag.destroy()
        self.ephemeris_track.close()
