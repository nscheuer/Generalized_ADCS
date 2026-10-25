"""Spacecraft model import: frames, units, mass properties, axis checks, LODs, scene link."""

import json
import struct

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("fast_simplification")

import ADCS  # noqa: E402
from ADCS.satellite_hardware.disturbances import (  # noqa: E402
    Drag_Disturbance,
    GeometryConfig,
    GeometryFace,
)
from ADCS.viz.mesh_import import PRESET_ROTATIONS, import_spacecraft  # noqa: E402
from ADCS.viz.mesh_import.__main__ import main as cli  # noqa: E402
from ADCS.viz.scene import build_scene  # noqa: E402

BUS = (0.10, 0.10, 0.20)
MASS = 2.0


def _rot(axis, deg):
    return trimesh.transformations.rotation_matrix(np.radians(deg), axis)


def _cubesat_scene(rw_tilt_deg=0.0, extra_transform=None):
    """2U bus with three wheels (inside), a star tracker looking along body +X."""
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.creation.box(extents=BUS), node_name="bus", geom_name="bus")
    # Cylinders are built along their local Z; rotate each onto its wheel axis.
    for name, T in (("rw_x", _rot([0, 1, 0], 90)), ("rw_y", _rot([1, 0, 0], -90)), ("rw_z", np.eye(4))):
        T = T.copy()
        if name == "rw_x" and rw_tilt_deg:
            T = _rot([0, 0, 1], rw_tilt_deg) @ T
        T[:3, 3] = {"rw_x": [0.0, 0.0, -0.05], "rw_y": [0.0, 0.0, 0.0], "rw_z": [0.0, 0.0, 0.05]}[name]
        sc.add_geometry(trimesh.creation.cylinder(radius=0.03, height=0.012, sections=48),
                        node_name=name, geom_name=name, transform=T)
    # Star tracker: local +Z boresight rotated onto body +X.
    T = _rot([0, 1, 0], 90)
    T[:3, 3] = [0.03, 0.0, 0.07]
    sc.add_geometry(trimesh.creation.box(extents=(0.02, 0.02, 0.03)), node_name="tracker",
                    geom_name="tracker", transform=T)
    if extra_transform is not None:
        sc.apply_transform(extra_transform)
    return sc


def _panels():
    a, b, c = BUS
    faces = []
    for axis, (area, half) in enumerate([(b * c, a / 2), (a * c, b / 2), (a * b, c / 2)]):
        for sign in (1.0, -1.0):
            n = np.zeros(3)
            n[axis] = sign
            faces.append(GeometryFace(area=area, centroid=n * half, normal=n, CD=2.2))
    return [Drag_Disturbance(GeometryConfig(faces))]


def _config(mesh, **kw):
    return {"name": "testsat", "mesh": str(mesh), "nodes": {
        "bus": {"role": "body"},
        "rw_x": {"role": "rw", "actuator": "rw0"}, "rw_y": {"role": "rw", "actuator": "rw1"},
        "rw_z": {"role": "rw", "actuator": "rw2"},
        "tracker": {"role": "sensor", "sensor": "startracker0"}}, **kw}


@pytest.fixture(scope="module")
def model_mass_properties(tmp_path_factory):
    """What the importer computes for the reference model (used to declare a matching satellite)."""
    d = tmp_path_factory.mktemp("ref")
    _cubesat_scene().export(d / "sat.glb")
    res = import_spacecraft(_config(d / "sat.glb", mass_kg=MASS), d / "out")
    assert res.ok, res.errors
    mp = json.loads(res.sidecar.read_text())["mass_properties"]["computed"]
    return np.array(mp["com_body_m"]), np.array(mp["inertia_com_kgm2"])


def _satellite(com, j_com):
    j0 = j_com + MASS * (np.dot(com, com) * np.eye(3) - np.outer(com, com))  # Satellite takes J about the origin
    rws = [ADCS.RW(axis=a, max_torque=0.01, J=1e-5, h=0.0, h_max=0.05) for a in np.eye(3)]
    return ADCS.Satellite(mass=MASS, COM=com, J_0=j0, actuators=rws, disturbances=_panels(),
                          sensors=[ADCS.StarTracker(boresight=np.array([1.0, 0.0, 0.0]))])


@pytest.fixture()
def sat(model_mass_properties):
    return _satellite(*model_mass_properties)


# ------------------------------------------------------------------ mass properties
def test_two_box_mass_properties_match_the_analytic_values(tmp_path):
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), node_name="a", geom_name="a",
                    transform=trimesh.transformations.translation_matrix([0, 0, -0.1]))
    sc.add_geometry(trimesh.creation.box(extents=(0.1, 0.1, 0.1)), node_name="b", geom_name="b",
                    transform=trimesh.transformations.translation_matrix([0, 0, 0.1]))
    sc.export(tmp_path / "two.glb")
    res = import_spacecraft({"mesh": str(tmp_path / "two.glb"), "mass_kg": 2.0}, tmp_path / "out")
    mp = json.loads(res.sidecar.read_text())["mass_properties"]["computed"]
    np.testing.assert_allclose(mp["com_body_m"], [0, 0, 0], atol=1e-12)
    each = 1.0 * 0.01 / 6.0  # m a^2 / 6 for a 1 kg, 10 cm cube
    expected = np.diag([2 * (each + 0.01), 2 * (each + 0.01), 2 * each])
    # glTF stores float32 vertices: ~1e-7 relative.
    np.testing.assert_allclose(mp["inertia_com_kgm2"], expected, rtol=1e-6, atol=1e-12)


def test_matching_model_imports_cleanly(tmp_path, sat):
    _cubesat_scene().export(tmp_path / "sat.glb")
    res = import_spacecraft(_config(tmp_path / "sat.glb"), tmp_path / "out", satellite=sat)
    assert res.ok and res.warnings == [], res.warnings
    side = json.loads(res.sidecar.read_text())
    assert side["coords"] == "BODY" and side["units"] == "m"
    assert side["nodes"]["rw_x"]["axis_err_deg"] < 0.1
    assert side["nodes"]["tracker"]["boresight_err_deg"] < 1e-6
    # Panel areas match, except +X: the tracker's front face lies 5 mm inside it,
    # within the 1 cm plane tolerance (4 cm^2 on 200 cm^2), below the 10% warning.
    assert max(p["rel_error"] for p in side["panels"]) == pytest.approx(0.02, abs=1e-3)
    assert sorted(p["rel_error"] for p in side["panels"])[-2] < 1e-6
    assert side["mass_properties"]["comparison"]["inertia_rel_error"] < 1e-9


def test_inertia_disagreement_is_reported(tmp_path, model_mass_properties):
    com, j = model_mass_properties
    sat = _satellite(com + np.array([0.0, 0.0, 0.03]), j * 1.5)
    _cubesat_scene().export(tmp_path / "sat.glb")
    res = import_spacecraft(_config(tmp_path / "sat.glb"), tmp_path / "out", satellite=sat)
    assert res.ok
    assert any("centre of mass" in w for w in res.warnings)
    assert any("inertia differs" in w for w in res.warnings)


# ------------------------------------------------------------------ frames and units
def test_stl_needs_units_and_wrong_units_are_flagged(tmp_path, sat):
    mesh = _cubesat_scene().to_mesh()
    mesh.apply_scale(1000.0)  # CAD export in millimetres
    mesh.export(tmp_path / "sat.stl")
    cfg = {"name": "stl", "mesh": str(tmp_path / "sat.stl")}
    res = import_spacecraft(cfg, tmp_path / "a", satellite=sat)
    assert not res.ok and "units" in res.errors[0]
    res = import_spacecraft({**cfg, "units": "m"}, tmp_path / "b", satellite=sat)
    assert any('try "units": "mm"' in w for w in res.warnings)
    res = import_spacecraft({**cfg, "units": "mm"}, tmp_path / "c", satellite=sat)
    assert not any("units" in w for w in res.warnings)


def test_y_up_export_is_undone_by_the_preset(tmp_path, sat):
    to_y_up = np.eye(4)
    to_y_up[:3, :3] = PRESET_ROTATIONS["gltf_y_up"].T  # what a Z-up -> Y-up exporter does
    _cubesat_scene(extra_transform=to_y_up).export(tmp_path / "yup.glb")
    res = import_spacecraft(_config(tmp_path / "yup.glb"), tmp_path / "raw", satellite=sat)
    assert any("wheel symmetry axis" in w for w in res.warnings)
    res = import_spacecraft(_config(tmp_path / "yup.glb", cad_to_body={"rotation": "gltf_y_up"}),
                            tmp_path / "fixed", satellite=sat)
    assert res.ok and res.warnings == [], res.warnings


def test_misaligned_wheel_and_sensor_are_flagged(tmp_path, sat):
    _cubesat_scene(rw_tilt_deg=10.0).export(tmp_path / "tilt.glb")
    cfg = _config(tmp_path / "tilt.glb")
    cfg["nodes"]["tracker"]["boresight_local"] = [1.0, 0.0, 0.0]
    res = import_spacecraft(cfg, tmp_path / "out", satellite=sat)
    assert any("rw_x" in w and "10.0 deg" in w for w in res.warnings)
    assert any("tracker" in w and "boresight" in w for w in res.warnings)


# ------------------------------------------------------------------ bad files
def test_unknown_ids_and_nodes_are_errors(tmp_path, sat):
    _cubesat_scene().export(tmp_path / "sat.glb")
    cfg = _config(tmp_path / "sat.glb")
    cfg["nodes"]["rw_x"]["actuator"] = "rw9"
    assert "no actuator 'rw9'" in import_spacecraft(cfg, tmp_path / "a", satellite=sat).errors[0]
    cfg = _config(tmp_path / "sat.glb")
    cfg["nodes"]["solar_array"] = {"role": "panel"}
    assert "not in the model" in import_spacecraft(cfg, tmp_path / "b", satellite=sat).errors[0]


def test_open_mesh_is_drawn_but_left_out_of_mass_properties(tmp_path):
    box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    open_box = trimesh.Trimesh(box.vertices, box.faces[2:], process=False)
    sc = trimesh.Scene()
    sc.add_geometry(box, node_name="closed", geom_name="closed")
    sc.add_geometry(open_box, node_name="open", geom_name="open",
                    transform=trimesh.transformations.translation_matrix([0.2, 0, 0]))
    sc.export(tmp_path / "m.glb")
    res = import_spacecraft({"mesh": str(tmp_path / "m.glb"), "mass_kg": 1.0}, tmp_path / "out")
    assert res.ok and any("'open' is not a closed volume" in w for w in res.warnings)
    mp = json.loads(res.sidecar.read_text())["mass_properties"]["computed"]
    assert mp["nodes_used"] == ["closed"]


def test_compressed_gltf_is_rejected(tmp_path):
    trimesh.creation.box().export(tmp_path / "m.glb")
    data = (tmp_path / "m.glb").read_bytes()
    jlen = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + jlen])
    doc["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    rest = data[20 + jlen:]
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + len(rest)) + struct.pack("<I4s", len(js), b"JSON") + js + rest
    (tmp_path / "draco.glb").write_bytes(out)
    res = import_spacecraft({"mesh": str(tmp_path / "draco.glb")}, tmp_path / "out")
    assert not res.ok and "KHR_draco_mesh_compression" in res.errors[0]


def test_non_uniform_node_scale_is_rejected(tmp_path):
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.creation.box(), node_name="stretched", geom_name="b",
                    transform=np.diag([1.0, 2.0, 1.0, 1.0]))
    sc.export(tmp_path / "m.glb")
    res = import_spacecraft({"mesh": str(tmp_path / "m.glb")}, tmp_path / "out")
    assert not res.ok and "non-uniform scale" in res.errors[0]


# ------------------------------------------------------------------ outputs
def test_levels_of_detail_keep_nodes_and_body_coordinates(tmp_path):
    sc = trimesh.Scene()
    sphere = trimesh.creation.icosphere(subdivisions=5, radius=0.2)  # 20480 triangles
    sc.add_geometry(sphere, node_name="dish", geom_name="dish",
                    transform=trimesh.transformations.translation_matrix([0.0, 0.0, 0.5]))
    sc.export(tmp_path / "m.glb")
    res = import_spacecraft({"name": "dishsat", "mesh": str(tmp_path / "m.glb"), "lods": [1.0, 0.25]},
                            tmp_path / "out")
    side = json.loads(res.sidecar.read_text())
    tri = [lod["triangles"] for lod in side["lods"]]
    assert tri[0] == 20480 and 0.2 * tri[0] < tri[1] < 0.3 * tri[0]
    for path in res.lods:
        data = path.read_bytes()
        doc = json.loads(data[20:20 + struct.unpack_from("<I", data, 12)[0]])
        assert "NORMAL" in doc["meshes"][0]["primitives"][0]["attributes"]  # viewers need them for lighting
        back = trimesh.load_scene(path)
        assert list(back.graph.nodes_geometry) == ["dish"]
        T, g = back.graph["dish"]
        np.testing.assert_allclose(T, np.eye(4))
        np.testing.assert_allclose(back.geometry[g].bounds.mean(axis=0), [0, 0, 0.5], atol=2e-3)


def test_scene_links_the_model(tmp_path, sat):
    _cubesat_scene().export(tmp_path / "sat.glb")
    res = import_spacecraft(_config(tmp_path / "sat.glb"), tmp_path / "out", satellite=sat)
    scene = build_scene(sat, model=res.sidecar)
    assert scene["mesh"]["coords"] == "BODY" and len(scene["mesh"]["lods"]) == 3
    assert {a["id"]: a["node"] for a in scene["actuators"]} == {"rw0": "rw_x", "rw1": "rw_y", "rw2": "rw_z"}
    assert scene["sensors"][0]["node"] == "tracker"


def test_scene_carries_node_roles_and_hinges(tmp_path, sat):
    _cubesat_scene().export(tmp_path / "sat.glb")
    cfg = _config(tmp_path / "sat.glb")
    cfg["nodes"]["bus"].update(articulation="hinge", axis_body=[0, 0, 1], pivot_body=[0.05, 0.05, 0.0])
    res = import_spacecraft(cfg, tmp_path / "out", satellite=sat)
    nodes = build_scene(sat, model=res.sidecar)["mesh"]["nodes"]
    assert nodes["rw_y"] == {"role": "rw", "actuator": "rw1", "articulation": "spin", "axis_body": [0.0, 1.0, 0.0]}
    assert nodes["bus"]["articulation"] == "hinge" and nodes["bus"]["pivot_body"] == [0.05, 0.05, 0.0]
    assert np.allclose(nodes["bus"]["axis_body"], [0, 0, 1])


def test_cli_exit_codes(tmp_path, sat, monkeypatch):
    _cubesat_scene(rw_tilt_deg=10.0).export(tmp_path / "tilt.glb")
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(_config("tilt.glb")))
    import sys
    import types
    mod = types.ModuleType("_test_sat_factory")
    mod.make = lambda: sat
    monkeypatch.setitem(sys.modules, "_test_sat_factory", mod)
    args = [str(cfg_path), "--out", str(tmp_path / "out"), "--satellite", "_test_sat_factory:make"]
    assert cli(args) == 0
    assert cli(args + ["--strict"]) == 1
    cfg_path.write_text(json.dumps({"mesh": "missing.glb"}))
    assert cli(args) == 2


def test_reduced_levels_of_textured_models_use_the_average_colour(tmp_path):
    from PIL import Image

    m = trimesh.creation.uv_sphere(radius=0.2, count=[96, 96])
    uv = np.column_stack([np.arctan2(m.vertices[:, 1], m.vertices[:, 0]) / (2 * np.pi) + 0.5,
                          0.5 + np.arcsin(np.clip(m.vertices[:, 2] / 0.2, -1, 1)) / np.pi])
    tex = np.zeros((8, 8, 3), dtype=np.uint8)
    tex[:, :4] = (200, 0, 0)
    tex[:, 4:] = (0, 0, 200)  # average (100, 0, 100)
    m.visual = trimesh.visual.TextureVisuals(
        uv=uv, material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex)))
    sc = trimesh.Scene()
    sc.add_geometry(m, node_name="tex", geom_name="tex")
    sc.export(tmp_path / "tex.glb")
    res = import_spacecraft({"name": "tex", "mesh": str(tmp_path / "tex.glb"), "lods": [1.0, 0.25]}, tmp_path / "out")
    full = list(trimesh.load_scene(res.lods[0]).geometry.values())[0]
    reduced = list(trimesh.load_scene(res.lods[1]).geometry.values())[0]
    assert full.visual.uv is not None  # full detail keeps the texture
    color = reduced.visual.main_color if hasattr(reduced.visual, "main_color") else reduced.visual.material.main_color
    np.testing.assert_allclose(np.asarray(color)[:3], [100, 0, 100], atol=2)


def test_flat_faces_get_hard_normals(tmp_path):
    sc = trimesh.Scene()
    sc.add_geometry(trimesh.creation.box(extents=(0.1, 0.1, 0.2)), node_name="bus", geom_name="bus")
    sc.export(tmp_path / "box.glb")
    res = import_spacecraft({"name": "box", "mesh": str(tmp_path / "box.glb"), "lods": [1.0]}, tmp_path / "out")
    m = list(trimesh.load_scene(res.lods[0]).geometry.values())[0]
    assert len(m.vertices) == 24  # 8 corners split per face
    face_n = m.face_normals[np.repeat(np.arange(len(m.faces)), 3)]
    vert_n = m.vertex_normals[m.faces.reshape(-1)]
    assert np.allclose((face_n * vert_n).sum(1), 1.0, atol=1e-5)


def test_crease_normals_match_trimesh_and_scale_to_cad_meshes():
    import time

    from ADCS.viz.mesh_import.importer import _shade

    cyl = trimesh.creation.cylinder(radius=1, height=2, sections=24)
    ours, ref = _shade(cyl), trimesh.graph.smooth_shade(cyl, angle=np.radians(30))
    assert len(ours.vertices) == len(ref.vertices) == 98  # same hard/smooth split: 2 cap rings + the side
    corner = ours.vertex_normals[ours.faces]
    assert np.einsum("fkj,fj->fk", corner, ours.face_normals).min() > np.cos(np.radians(30))
    # A finely tessellated sphere (~330k faces) in seconds: trimesh's version takes minutes at this size.
    big = trimesh.creation.icosphere(subdivisions=7)
    t0 = time.perf_counter()
    shaded = _shade(big)
    assert time.perf_counter() - t0 < 30.0
    assert len(shaded.vertices) == len(big.vertices)  # all smooth: no splits
