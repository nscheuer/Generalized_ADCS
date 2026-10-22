"""Reusable static geometry for Panda3D scene objects."""

import math


def uv_sphere(radius: float, slices: int = 128, stacks: int = 64):
    """Build one indexed, UV mapped sphere with outward normals and no frame updates."""
    from panda3d.core import (
        Geom, GeomNode, GeomTriangles, GeomVertexData, GeomVertexFormat,
        GeomVertexWriter,
    )

    data = GeomVertexData("uv sphere", GeomVertexFormat.getV3n3t2(), Geom.UHStatic)
    vertex = GeomVertexWriter(data, "vertex")
    normal = GeomVertexWriter(data, "normal")
    uv = GeomVertexWriter(data, "texcoord")
    for row in range(stacks + 1):
        latitude = math.pi * (0.5 - row / stacks)
        z, ring = math.sin(latitude), math.cos(latitude)
        for col in range(slices + 1):
            longitude = 2 * math.pi * col / slices
            x, y = ring * math.cos(longitude), ring * math.sin(longitude)
            vertex.addData3f(radius * x, radius * y, radius * z)
            normal.addData3f(x, y, z)
            uv.addData2f(col / slices, 1 - row / stacks)

    triangles = GeomTriangles(Geom.UHStatic)
    for row in range(stacks):
        for col in range(slices):
            a = row * (slices + 1) + col
            b = a + slices + 1
            triangles.addVertices(a, b, a + 1)
            triangles.addVertices(a + 1, b, b + 1)
    geom = Geom(data)
    geom.addPrimitive(triangles)
    node = GeomNode("uv sphere")
    node.addGeom(geom)
    return node
