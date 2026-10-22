"""Textured Saturn ring annulus with the planet's equatorial orientation."""

from math import cos, pi, sin
from pathlib import Path
from ._texture_assets import texture_path, tier_name


RING_TEXTURE = (
    Path(__file__).resolve().parents[1]
    / 'plotting' / 'textures' / '2k_saturn_ring_alpha.png'
)


def _annulus_geometry(slices: int = 384):
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter,
    )

    # Saturn's main rings extend from the inner D ring to the outer A ring.
    inner_radius, outer_radius = 1.11, 2.27
    data = GeomVertexData('Saturn ring annulus', GeomVertexFormat.getV3n3t2(),
                          Geom.UHStatic)
    vertex = GeomVertexWriter(data, 'vertex')
    normal = GeomVertexWriter(data, 'normal')
    uv = GeomVertexWriter(data, 'texcoord')

    for index in range(slices + 1):
        angle = 2.0 * pi * index / slices
        x, y = cos(angle), sin(angle)
        angular_uv = index / slices
        vertex.addData3f(inner_radius * x, inner_radius * y, 0)
        normal.addData3f(0, 0, 1)
        uv.addData2f(0, angular_uv)
        vertex.addData3f(outer_radius * x, outer_radius * y, 0)
        normal.addData3f(0, 0, 1)
        uv.addData2f(1, angular_uv)

    triangles = GeomTriangles(Geom.UHStatic)
    for index in range(slices):
        inner = 2 * index
        outer = inner + 1
        next_inner = inner + 2
        next_outer = inner + 3
        triangles.addVertices(inner, outer, next_outer)
        triangles.addVertices(inner, next_outer, next_inner)

    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode('Saturn rings')
    node.addGeom(geom)
    return node


class SaturnRings:
    """Manage Saturn's rings independently from its spherical body LOD."""

    def __init__(self, base, parent, planet_radius_km: float, tier='medium') -> None:
        from panda3d.core import TransparencyAttrib

        self.base = base
        self.texture = None
        self.tier = tier
        self.node = parent.attachNewNode(_annulus_geometry())
        self.node.setScale(planet_radius_km)
        self.node.setColor(1, 1, 1, 1)
        self.node.setTwoSided(True)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setDepthTest(True)
        self.node.setDepthWrite(False)
        self.node.setBin('transparent', 0)
        self.node.hide()

    def set_orientation(self, quat) -> None:
        self.node.setQuat(quat)

    def set_visible(self, visible: bool) -> None:
        if visible and self.texture is None:
            path = texture_path(tier_name(self.tier, '2k_saturn_ring_alpha.png'))
            texture = self.base.loader.loadTexture(str(path))
            if texture is None:
                return
            from panda3d.core import Texture
            texture.setMinfilter(Texture.FTLinear)
            texture.setMagfilter(Texture.FTLinear)
            self.texture = texture
            self.node.setTexture(texture, 1)
        if visible and self.texture is not None:
            self.node.show()
        else:
            self.node.hide()

    def set_quality(self, tier):
        was_visible = not self.node.isHidden()
        self.node.clearTexture()
        if self.texture is not None:
            self.base.loader.unloadTexture(self.texture)
            self.texture = None
        self.tier = tier
        self.set_visible(was_visible)

    def close(self) -> None:
        self.node.removeNode()
        if self.texture is not None:
            self.base.loader.unloadTexture(self.texture)
            self.texture = None
