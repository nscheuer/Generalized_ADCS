"""Optional NASA spacecraft models for the Godot orbit viewer."""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

from ADCS.helpers.asset_manager import Asset, AssetManager

from .fetch import cache_dir


NASA_ASSETS = "https://assets.science.nasa.gov/content/dam/science/cds/3d/resources/model/"
MODELS = {
    "cloudsat": ("CloudSat (C)", "cloudsat-(c)/CloudSat%20(C).glb"),
    "cubesat_1ru": ("CubeSat 1 RU", "cubesat---1-ru-generic/CubeSat%20-%201%20RU%20Generic.glb"),
    "cubesat_2ru": ("CubeSat 2 RU", "cubesat---2-ru-generic/CubeSat%20-%202%20RU%20Generic.glb"),
    "icecube": ("CubeSat - ICECube", "cubesat---icecube/CubeSat%20-%20ICECube.glb"),
    "mirata": ("CubeSat MiRaTa", "cubesat---mirata/CubeSat%20-%20MiRaTa.glb"),
    "icesat2": ("ICESat-2", "ice,-clouds,-and-land-elevation-satellite-2-(icesat-2)-(b)/Ice,%20Clouds,%20and%20Land%20Elevation%20Satellite-2%20(ICESat-2)%20(B).glb"),
}


def model_path(name: str, root: Path | None = None) -> Path:
    if name not in MODELS:
        raise ValueError(f"unknown NASA spacecraft model: {name}")
    return (root or cache_dir()) / "spacecraft_models" / f"{name}.glb"


def godot_model_path(name: str, root: Path | None = None) -> Path:
    """Cached Godot-compatible copy; the original NASA download stays intact."""
    return model_path(name, root).with_name(f"{name}.godot.glb")


def ensure_model(name: str, *, root: Path | None = None) -> Path | None:
    """Return a cached Godot-compatible model, asking before fetching it."""
    if name not in MODELS:
        raise ValueError(f"unknown NASA spacecraft model: {name}")
    root = root or cache_dir()
    target = model_path(name, root)
    compatible = godot_model_path(name, root)
    if compatible.is_file() and (not target.is_file() or compatible.stat().st_mtime_ns >= target.stat().st_mtime_ns):
        return compatible
    if not target.is_file():
        label, url_path = MODELS[name]
        url = NASA_ASSETS + url_path
        asset = Asset(name, url, target.name)
        manager = AssetManager(registry={name: asset}, locations=[], cache_dir=target.parent)
        if not sys.stdin.isatty():
            print(f"Missing NASA model {label}. Run the orbit viewer in a terminal to approve its download.", flush=True)
            return None
        try:
            manager.get(name, destination=target, interactive=True)
        except FileNotFoundError as error:
            print(f"{error}; keeping the current spacecraft model.", flush=True)
            return None
    return _make_godot_compatible(target, compatible)


def _make_godot_compatible(source: Path, destination: Path) -> Path:
    """Expand NASA's Draco-compressed GLB primitives into regular glTF buffers."""
    data = source.read_bytes()
    if data[:4] != b"glTF":
        raise ValueError(f"{source} is not a binary glTF file")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2 or total_length > len(data):
        raise ValueError(f"{source} has an invalid GLB header")

    chunks = []
    offset = 12
    while offset < total_length:
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        chunks.append((kind, data[offset:offset + length]))
        offset += length
    json_chunk = next((chunk for kind, chunk in chunks if kind == 0x4E4F534A), None)
    binary_chunk = next((chunk for kind, chunk in chunks if kind == 0x004E4942), b"")
    if json_chunk is None:
        raise ValueError(f"{source} has no glTF JSON chunk")
    document = json.loads(json_chunk.decode("utf-8").rstrip(" \t\r\n\0"))
    compressed_primitives = [
        primitive
        for mesh in document.get("meshes", [])
        for primitive in mesh.get("primitives", [])
        if "KHR_draco_mesh_compression" in primitive.get("extensions", {})
    ]
    if not compressed_primitives:
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        return destination

    try:
        import numpy as np
        import DracoPy
    except ImportError as error:
        raise RuntimeError("NASA spacecraft GLBs need DracoPy to convert for Godot; install the project dependencies again") from error

    binary = bytearray(binary_chunk)
    buffer_views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    component_types = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
                       5125: np.uint32, 5126: np.float32}

    def append_view(payload: bytes, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        view_index = len(buffer_views)
        buffer_views.append({"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload), "target": target})
        binary.extend(payload)
        return view_index

    for primitive in compressed_primitives:
        compression = primitive["extensions"].pop("KHR_draco_mesh_compression")
        if not primitive["extensions"]:
            primitive.pop("extensions")
        view = buffer_views[compression["bufferView"]]
        start = int(view.get("byteOffset", 0))
        compressed = binary_chunk[start:start + int(view["byteLength"])]
        decoded = DracoPy.decode(compressed)

        for semantic, accessor_index in primitive["attributes"].items():
            attribute_id = compression["attributes"].get(semantic)
            if attribute_id is None:
                continue
            attribute = decoded.get_attribute_by_unique_id(int(attribute_id))
            accessor = accessors[int(accessor_index)]
            dtype = component_types.get(int(accessor["componentType"]))
            if dtype is None:
                raise ValueError(f"Unsupported glTF component type in {source}: {accessor['componentType']}")
            values = np.asarray(attribute["data"], dtype=dtype)
            accessor["bufferView"] = append_view(values.tobytes(order="C"), 34962)
            accessor["byteOffset"] = 0
            accessor["count"] = int(values.shape[0])

        if "indices" in primitive:
            accessor = accessors[int(primitive["indices"])]
            dtype = component_types.get(int(accessor["componentType"]))
            if dtype not in (np.uint8, np.uint16, np.uint32):
                raise ValueError(f"Unsupported glTF index type in {source}: {accessor['componentType']}")
            indices = np.asarray(decoded.faces, dtype=dtype).reshape(-1)
            accessor["bufferView"] = append_view(indices.tobytes(order="C"), 34963)
            accessor["byteOffset"] = 0
            accessor["count"] = int(indices.size)

    for extension_list in ("extensionsUsed", "extensionsRequired"):
        if extension_list in document:
            document[extension_list] = [ext for ext in document[extension_list] if ext != "KHR_draco_mesh_compression"]
            if not document[extension_list]:
                document.pop(extension_list)
    document["buffers"][0]["byteLength"] = len(binary)
    json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    binary.extend(b"\0" * ((-len(binary)) % 4))
    output = bytearray(b"glTF")
    output.extend(struct.pack("<II", 2, 0))
    output.extend(struct.pack("<II", len(json_bytes), 0x4E4F534A))
    output.extend(json_bytes)
    output.extend(struct.pack("<II", len(binary), 0x004E4942))
    output.extend(binary)
    struct.pack_into("<I", output, 8, len(output))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(output)
    temporary.replace(destination)
    return destination
