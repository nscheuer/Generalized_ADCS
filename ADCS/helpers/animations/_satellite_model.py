"""NASA CubeSat model cache and recorded spacecraft pose."""

from pathlib import Path
import json
import struct
from urllib.request import Request, urlopen

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, RotationSpline

from ._texture_assets import _cache_dir


MODEL_URLS = {
    '1ru': 'https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/cubesat---1-ru-generic/CubeSat%20-%201%20RU%20Generic.glb',
    '2ru': 'https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/cubesat---2-ru-generic/CubeSat%20-%202%20RU%20Generic.glb',
    'icecube': 'https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/cubesat---icecube/CubeSat%20-%20ICECube.glb',
    'mirata': 'https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/cubesat---mirata/CubeSat%20-%20MiRaTa.glb',
}


def _load_draco_glb(parent, path):
    """Decode NASA's Draco GLB meshes into Panda geometry once per viewer."""
    try:
        import DracoPy
    except ImportError as exc:
        raise RuntimeError('CubeSat models require DracoPy; install generalized-adcs[panda3d]') from exc
    from panda3d.core import (Geom, GeomNode, GeomTriangles, GeomVertexData,
                              GeomVertexFormat, GeomVertexWriter)

    data = path.read_bytes()
    if data[:4] != b'glTF':
        raise ValueError(f'Invalid GLB: {path}')
    json_size = struct.unpack_from('<I', data, 12)[0]
    document = json.loads(data[20:20 + json_size])
    chunk_start = 20 + json_size
    binary_size = struct.unpack_from('<I', data, chunk_start)[0]
    binary = data[chunk_start + 8:chunk_start + 8 + binary_size]
    # glTF is Y-up; Panda's world is Z-up. This rotation preserves handedness.
    conversion = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)

    def node_transform(node):
        if 'matrix' in node:
            return np.asarray(node['matrix'], dtype=float).reshape((4, 4), order='F')
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(node.get('rotation', [0, 0, 0, 1])).as_matrix()
        transform[:3, :3] *= np.asarray(node.get('scale', [1, 1, 1]), dtype=float)
        transform[:3, 3] = node.get('translation', [0, 0, 0])
        return transform

    def visit(index, parent_transform):
        node = document['nodes'][index]
        transform = parent_transform @ node_transform(node)
        if 'mesh' in node:
            mesh = document['meshes'][node['mesh']]
            for part, primitive in enumerate(mesh['primitives']):
                extension = primitive.get('extensions', {}).get('KHR_draco_mesh_compression')
                if extension is None:
                    raise ValueError('NASA CubeSat GLB has an unsupported mesh primitive')
                view = document['bufferViews'][extension['bufferView']]
                offset = view.get('byteOffset', 0)
                decoded = DracoPy.decode(binary[offset:offset + view['byteLength']])
                points = np.asarray(decoded.points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]
                points = points @ conversion.T
                normals = np.asarray(decoded.normals, dtype=float) @ np.linalg.inv(transform[:3, :3])
                normals = normals @ conversion.T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
                vertices = GeomVertexData('CubeSat', GeomVertexFormat.getV3n3(), Geom.UHStatic)
                vertices.setNumRows(len(points))
                xyz = GeomVertexWriter(vertices, 'vertex')
                normal = GeomVertexWriter(vertices, 'normal')
                for point, direction in zip(points, normals):
                    xyz.addData3f(*point)
                    normal.addData3f(*direction)
                triangles = GeomTriangles(Geom.UHStatic)
                for face in decoded.faces:
                    triangles.addVertices(*(int(i) for i in face))
                geom = Geom(vertices)
                geom.addPrimitive(triangles)
                geom_node = GeomNode(f'CubeSat {index}:{part}')
                geom_node.addGeom(geom)
                instance = parent.attachNewNode(geom_node)
                color = document['materials'][primitive['material']].get('pbrMetallicRoughness', {}).get(
                    'baseColorFactor', [0.75, 0.8, 0.85, 1])
                instance.setColor(*color)
                instance.setTwoSided(True)
                instance.setLightOff()
        for child in node.get('children', []):
            visit(child, transform)

    scene = document['scenes'][document.get('scene', 0)]
    for index in scene['nodes']:
        visit(index, np.eye(4))
    return parent


def model_path(name, on_progress=None):
    name = name.lower()
    if name not in MODEL_URLS:
        raise ValueError(f"satellite_model must be one of {', '.join(MODEL_URLS)}")
    target = _cache_dir() / 'models' / f'cubesat_{name}.glb'
    if target.is_file() and target.stat().st_size:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.glb.part')
    try:
        request = Request(MODEL_URLS[name], headers={'User-Agent': 'Generalized_ADCS/0.2'})
        with urlopen(request, timeout=30) as source, temporary.open('wb') as output:
            total = int(source.headers.get('Content-Length') or 0)
            done = 0
            while block := source.read(64 * 1024):
                output.write(block)
                done += len(block)
                if on_progress is not None:
                    on_progress(done, total or done, f'Downloading {name} CubeSat')
            if total and done != total:
                raise IOError(f'Incomplete CubeSat download: {done} of {total} bytes')
        # GLB magic guards against caching an HTML error response.
        with temporary.open('rb') as model:
            if model.read(4) != b'glTF':
                raise IOError(f'NASA did not return a GLB model for {name}')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


class SatelliteModel:
    # The NASA 1RU bus spans about 2.2 model units and is 0.1 m wide.
    KM_PER_MODEL_UNIT = 0.0001 / 2.2

    def __init__(self, base, path: Path, time_hist, state_hist, os_hist, earth_root):
        self.base = base
        self.times = np.asarray(time_hist, dtype=float)
        self.positions = np.asarray([orb.R for orb in os_hist], dtype=float)
        self.velocities = np.asarray([orb.V for orb in os_hist], dtype=float)
        self.position_spline = (CubicHermiteSpline(self.times, self.positions, self.velocities)
                                if len(self.times) > 1 else None)
        quaternions = np.asarray([state.q for state in state_hist], dtype=float)
        quaternions /= np.linalg.norm(quaternions, axis=1)[:, None]
        rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
        self.rotation_spline = RotationSpline(self.times, rotations) if len(self.times) > 1 else None
        self.constant_rotation = rotations[0]
        self.absolute_position = np.zeros(3)
        self.root = base.render.attachNewNode('Spacecraft')
        self.model = self.root.attachNewNode('CubeSat geometry')
        try:
            _load_draco_glb(self.model, path)
        except Exception:
            self.root.removeNode()
            raise
        bounds = self.model.getTightBounds()
        if bounds is not None:
            minimum, maximum = bounds
            center = (0.5 * (minimum.x + maximum.x),
                      0.5 * (minimum.y + maximum.y),
                      0.5 * (minimum.z + maximum.z))
            self.model_radius = max((maximum - minimum).length() * 0.5, 1e-6)
        else:
            center = (0.0, 0.0, 0.0)
            self.model_radius = 1.0
        self.model.setScale(self.KM_PER_MODEL_UNIT)
        self.model.setPos(*(-value * self.KM_PER_MODEL_UNIT for value in center))
        self.radius_km = self.model_radius * self.KM_PER_MODEL_UNIT

        from panda3d.core import CardMaker, LineSegs, TransparencyAttrib

        maker = CardMaker('spacecraft marker')
        maker.setFrame(-1, 1, -1, 1)
        self.marker = base.aspect2d.attachNewNode(maker.generate())
        self.marker.setColor(0.35, 0.95, 1.0, 1.0)
        self.marker.setTransparency(TransparencyAttrib.MAlpha)
        self.marker.setDepthTest(False)
        self.marker.setDepthWrite(False)
        self.marker.setBin('fixed', 2)
        self.marker.hide()
        self.model.setTransparency(TransparencyAttrib.MAlpha)

        segments = LineSegs('recorded spacecraft orbit')
        segments.setThickness(2.0)
        segments.setColor(0.35, 0.95, 1.0, 1.0)
        # Draw the same curved trajectory used for playback, so sparse orbit
        # histories do not leave a visibly polygonal path behind the craft.
        if self.position_spline is None:
            trail_points = self.positions
        else:
            count = min(8192, max(len(self.times), 8 * (len(self.times) - 1) + 1))
            trail_points = self.position_spline(np.linspace(self.times[0], self.times[-1], count))
        segments.moveTo(*trail_points[0])
        for point in trail_points[1:]:
            segments.drawTo(*point)
        self.trail = earth_root.attachNewNode(segments.create())
        self.trail.setLightOff()
        self.trail.setDepthTest(True)
        self.trail.setDepthWrite(False)
        self.trail.setTransparency(TransparencyAttrib.MAlpha)
        self.trail.setBin('transparent', 1)
        self.trail.hide()

    def update(self, simulation_time, earth_position, camera_position,
               center_on_satellite=False):
        if simulation_time is None:
            self.root.hide()
            self.marker.hide()
            self.trail.hide()
            return
        time = float(np.clip(simulation_time, self.times[0], self.times[-1]))
        position = self.position_spline(time) if self.position_spline is not None else self.positions[0]
        self.absolute_position = np.asarray(tuple(earth_position), dtype=float) + position
        # Keep the tiny spacecraft near render's origin while it is focused.
        # Cancelling two ~7000 km float32 transforms otherwise moves a 10 cm
        # mesh by metres between frames.
        self.root.setPos(*(np.zeros(3) if center_on_satellite else self.absolute_position))
        rotation = (self.rotation_spline(time) if self.rotation_spline is not None
                    else self.constant_rotation)
        x, y, z, w = rotation.as_quat()
        from panda3d.core import Quat
        self.root.setQuat(Quat(w, x, y, z))
        distance = (camera_position - self.root.getPos(self.base.render)).length()
        self.trail.show() if distance > max(0.01, 100.0 * self.radius_km) else self.trail.hide()
        self.root.show()

    def update_marker(self):
        """Crossfade the model into a screen-space marker as it shrinks."""
        if self.root.isHidden():
            self.marker.hide()
            return
        from panda3d.core import LPoint2f
        camera_point = self.base.camera.getRelativePoint(
            self.base.render, self.root.getPos(self.base.render))
        projected = LPoint2f()
        if camera_point.y <= 0 or not self.base.camLens.project(camera_point, projected):
            self.marker.hide()
            return
        region = self.base.camNode.getDisplayRegion(0)
        focal = region.getPixelHeight() / (2 * np.tan(np.deg2rad(self.base.camLens.getVfov()) / 2))
        diameter = 2 * self.radius_km * focal / camera_point.y
        transition = np.clip((diameter - 2.0) / 6.0, 0.0, 1.0)
        model_alpha = float(transition * transition * (3.0 - 2.0 * transition))
        self.model.setColorScale(1, 1, 1, model_alpha)
        if model_alpha >= 1.0 or abs(projected.x) > 1 or abs(projected.y) > 1:
            self.marker.hide()
            return
        self.marker.setColorScale(1, 1, 1, 1.0 - model_alpha)
        self.marker.setScale(6.0 / max(self.base.win.getYSize(), 1))
        self.marker.setPos(projected.x * self.base.a2dRight, 0,
                           (region.getBottom() + (1 - region.getBottom()) * projected.y)
                           * self.base.a2dTop)
        self.marker.show()

    def pick(self, mouse):
        if mouse is None or self.root.isHidden():
            return False
        from panda3d.core import LPoint2f
        camera_point = self.base.camera.getRelativePoint(self.base.render,
                                                          self.root.getPos(self.base.render))
        projected = LPoint2f()
        if camera_point.y <= 0 or not self.base.camLens.project(camera_point, projected):
            return False
        region = self.base.camNode.getDisplayRegion(0)
        window_height = self.base.win.getYSize()
        center_x = projected.x * self.base.a2dRight
        center_y = (region.getBottom() + (1 - region.getBottom()) * projected.y) * self.base.a2dTop
        dx = (float(mouse.x) * self.base.a2dRight - center_x) * window_height / 2
        dy = (float(mouse.y) * self.base.a2dTop - center_y) * window_height / 2
        focal = region.getPixelHeight() / (2 * np.tan(np.deg2rad(self.base.camLens.getVfov()) / 2))
        radius = max(10.0, self.radius_km * focal / max(camera_point.length(), 1e-9))
        return dx * dx + dy * dy <= radius * radius

    def close(self):
        self.marker.removeNode()
        self.trail.removeNode()
        self.root.removeNode()
