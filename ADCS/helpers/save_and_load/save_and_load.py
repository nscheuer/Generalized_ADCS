from __future__ import annotations
__all__ = ["save_data", "load_data", "load_orbital_states"]

import json
import pickle
import sys
import inspect
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

def _now_stamp() -> str:
    """Return timestamp string, e.g. 20260107_235959"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _is_numeric_ndarray(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.dtype != object


def _looks_like_orbital_state(x: Any) -> bool:
    """Heuristic: do not import Orbital_State here"""
    return hasattr(x, "R") and hasattr(x, "V") and hasattr(x, "J2000")


def _is_orbital_state_list(x: Any) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False
    return _looks_like_orbital_state(x[0])


def _infer_var_name(obj: Any) -> Optional[str]:
    """
    Best-effort inference of the caller's variable name.
    Works for literal variables, not expressions.
    """
    try:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None:
            return None
        caller = frame.f_back.f_back
        if caller is None:
            return None

        for scope in (caller.f_locals, caller.f_globals):
            for k, v in scope.items():
                if v is obj:
                    return k
    except Exception:
        return None
    finally:
        try:
            del frame
        except Exception:
            pass
    return None


def _describe_obj(obj: Any) -> Dict[str, Any]:
    """Lightweight metadata for manifest.json"""
    info: Dict[str, Any] = {"type": type(obj).__name__}

    if isinstance(obj, np.ndarray):
        info.update(
            shape=list(obj.shape),
            dtype=str(obj.dtype),
            ndim=int(obj.ndim),
            size=int(obj.size),
        )
        return info

    if isinstance(obj, (list, tuple)):
        info["len"] = len(obj)
        if len(obj) > 0:
            info["elem_type"] = type(obj[0]).__name__
            if isinstance(obj[0], np.ndarray):
                info["elem_shape"] = list(obj[0].shape)
                info["elem_dtype"] = str(obj[0].dtype)
            if _looks_like_orbital_state(obj[0]):
                try:
                    info["elem_R_shape"] = list(np.asarray(obj[0].R).shape)
                    info["elem_V_shape"] = list(np.asarray(obj[0].V).shape)
                except Exception:
                    pass
        return info

    return info


@dataclass
class _ManifestItem:
    label: str
    var_name: str
    kind: str                 # "ndarray" | "pickle" | "orbital_state_list"
    file: str
    key: str
    info: Dict[str, Any]


def save_data(
    name: str | None = None,
    *objs: Any,
    labels: Optional[Iterable[str]] = None,
    out_dir: str | Path = "output",
    path: str | Path | None = None,
    add_timestamp: bool = True,
    compress: bool = True,
) -> str:
    r"""
    Persist heterogeneous simulation data to disk with automatic type handling.

    This function saves an arbitrary collection of Python objects into a
    structured output directory. Objects are grouped and serialized based on
    their detected type to support efficient storage, inspection, and later
    reconstruction.

    ======================
    Storage Strategy
    ======================

    Each input object is categorized into one of the following classes:

    +------------------------+---------------------------------------------+
    | Category               | Storage Method                              |
    +========================+=============================================+
    | Numeric ``ndarray``    | ``arrays.npz`` (NumPy compressed archive)   |
    +------------------------+---------------------------------------------+
    | ``list[Orbital_State]``| ``orbital_states.pkl`` (dict representation)|
    +------------------------+---------------------------------------------+
    | Other Python objects   | ``objects.pkl`` (pickle)                    |
    +------------------------+---------------------------------------------+

    A machine-readable ``manifest.json`` file is always generated. It records
    metadata for each saved object, including:

    * Original variable name (best-effort inference)
    * Storage file and key
    * Object shape, type, and basic structural information

    ======================
    Directory Naming
    ======================

    The output directory is determined as follows:

    * If ``path`` is provided, it is used directly
    * Otherwise, a directory named ``<name>[_YYYYMMDD_HHMMSS]`` is created
      inside ``out_dir``

    ======================
    Design Notes
    ======================

    * Orbital states are stored via ``to_dict()`` to avoid import-time coupling
    * Compression is applied only to NumPy archives
    * The function fails fast if directory collisions occur

    :param name:
        Base name of the output directory (ignored if ``path`` is provided).
    :type name:
        str or None

    :param objs:
        Arbitrary Python objects to be saved.
    :type objs:
        Any

    :param labels:
        Optional iterable of string labels corresponding one-to-one with
        ``objs``.
    :type labels:
        iterable of str or None

    :param out_dir:
        Parent directory under which the output directory is created.
    :type out_dir:
        str or pathlib.Path

    :param path:
        Explicit path to the output directory. Overrides ``name`` and
        ``out_dir``.
    :type path:
        str or pathlib.Path or None

    :param add_timestamp:
        If True, append a timestamp suffix to the directory name.
    :type add_timestamp:
        bool

    :param compress:
        If True, use compressed NumPy archives for arrays.
    :type compress:
        bool

    :return:
        Absolute path to the created output directory.
    :rtype:
        str

    """
    out_dir = Path(out_dir)

    if path is not None:
        run_dir = Path(path)
    else:
        if name is None:
            raise ValueError("Either `name` or `path` must be provided")
        suffix = f"_{_now_stamp()}" if add_timestamp else ""
        run_dir = out_dir / f"{name}{suffix}"

    run_dir = run_dir.resolve()
    print(f"[save_data] Saving to: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    if labels is None:
        labels_list = [f"obj{i}" for i in range(len(objs))]
    else:
        labels_list = list(labels)

    if len(labels_list) != len(objs):
        raise ValueError("labels must match number of objects")

    arrays: Dict[str, np.ndarray] = {}
    objects: Dict[str, Any] = {}
    orbital_states: Dict[str, List[dict]] = {}
    manifest: List[_ManifestItem] = []

    for label, obj in zip(labels_list, objs):
        var_name = _infer_var_name(obj) or label
        info = _describe_obj(obj)

        if _is_numeric_ndarray(obj):
            arrays[label] = obj
            kind, file = "ndarray", "arrays.npz"

        elif _is_orbital_state_list(obj):
            orbital_states[label] = [os.to_dict() for os in obj]
            kind, file = "orbital_state_list", "orbital_states.pkl"

        else:
            objects[label] = obj
            kind, file = "pickle", "objects.pkl"

        manifest.append(
            _ManifestItem(
                label=label,
                var_name=var_name,
                kind=kind,
                file=file,
                key=label,
                info=info,
            )
        )

    if arrays:
        fn = run_dir / "arrays.npz"
        np.savez_compressed(fn, **arrays) if compress else np.savez(fn, **arrays)

    if objects:
        with open(run_dir / "objects.pkl", "wb") as f:
            pickle.dump(objects, f, protocol=pickle.HIGHEST_PROTOCOL)

    if orbital_states:
        with open(run_dir / "orbital_states.pkl", "wb") as f:
            pickle.dump(orbital_states, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created": datetime.now().isoformat(timespec="seconds"),
                "python": sys.version,
                "numpy": np.__version__,
                "directory": str(run_dir),
                "items": [asdict(m) for m in manifest],
            },
            f,
            indent=2,
        )

    print(f"[save_data] Done.")
    return str(run_dir)

def load_data(run_dir: str | Path) -> Tuple[Any, ...]:
    r"""
    Load previously saved data exactly as written by :func:`save_data`.

    This function reconstructs objects from a saved run directory by reading
    the ``manifest.json`` file and loading each item from its corresponding
    storage file. Objects are returned in the same order in which they were
    originally saved.

    ======================
    Reconstruction Rules
    ======================

    * NumPy arrays are loaded from ``arrays.npz``
    * Pickled Python objects are loaded from ``objects.pkl``
    * Orbital state lists are returned as ``list[dict]`` without reconstruction

    This function performs no type inference or object reconstruction beyond
    basic deserialization.

    ======================
    Ordering Guarantee
    ======================

    The return order strictly follows the item order recorded in the manifest.
    This ensures positional consistency even when multiple objects share the
    same storage file.

    :param run_dir:
        Path to the directory created by :func:`save_data`.
    :type run_dir:
        str or pathlib.Path

    :return:
        Tuple of loaded objects in the original save order.
    :rtype:
        tuple

    """
    run_dir = Path(run_dir).resolve()
    print(f"[load_data] Loading from: {run_dir}")

    with open(run_dir / "manifest.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = None
    objects_dict = None
    orbital_dict = None

    out: List[Any] = []
    for it in meta["items"]:
        kind, file, key = it["kind"], it["file"], it["key"]

        if kind == "ndarray":
            if arrays is None:
                arrays = np.load(run_dir / file, allow_pickle=False)
            out.append(arrays[key])

        elif kind == "pickle":
            if objects_dict is None:
                with open(run_dir / file, "rb") as f:
                    objects_dict = pickle.load(f)
            out.append(objects_dict[key])

        elif kind == "orbital_state_list":
            if orbital_dict is None:
                with open(run_dir / file, "rb") as f:
                    orbital_dict = pickle.load(f)
            out.append(orbital_dict[key])

        else:
            raise ValueError(f"Unknown kind: {kind}")

    print(f"[load_data] Done.")
    return tuple(out)


def load_orbital_states(
    run_dir: str | Path,
    label: Optional[str] = None,
    ephem=None,
    density_model=None,
    fast: bool = True,
):
    r"""
    Load and optionally reconstruct orbital state histories.

    This function loads orbital state data stored by :func:`save_data` and
    provides two modes of operation:

    * **Raw mode**: return ``list[dict]`` representations
    * **Reconstruction mode**: return fully instantiated
      :class:`~ADCS.orbits.orbital_state.Orbital_State` objects

    ======================
    Reconstruction Logic
    ======================

    If an ephemeris object is provided, each dictionary entry is reconstructed
    using:

    .. code-block:: python

        Orbital_State.from_dict(...)

    This allows deferred coupling to orbital dynamics models and accelerates
    I/O when reconstruction is not required.

    ======================
    Label Resolution
    ======================

    * If ``label`` is provided, only that orbital state list is returned
    * If exactly one orbital state list exists, it is returned directly
    * Otherwise, a dictionary mapping labels to lists is returned

    ======================
    Performance Notes
    ======================

    The ``fast`` flag is passed directly to the orbital state constructor and
    may disable expensive precomputations.

    :param run_dir:
        Path to the directory created by :func:`save_data`.
    :type run_dir:
        str or pathlib.Path

    :param label:
        Optional label identifying a specific orbital state list.
    :type label:
        str or None

    :param ephem:
        Ephemeris object required to reconstruct
        :class:`~ADCS.orbits.orbital_state.Orbital_State` instances.
    :type ephem:
        Any or None

    :param density_model:
        Optional atmospheric density model passed during reconstruction.
    :type density_model:
        Any or None

    :param fast:
        If True, enable fast reconstruction mode.
    :type fast:
        bool

    :return:
        Orbital state history in raw or reconstructed form.
    :rtype:
        list or dict

    """
    run_dir = Path(run_dir).resolve()
    print(f"[load_orbital_states] Loading from: {run_dir}")

    with open(run_dir / "orbital_states.pkl", "rb") as f:
        d = pickle.load(f)

    def reconstruct(lst):
        if ephem is None:
            return lst
        from ADCS.orbits.orbital_state import Orbital_State
        return [
            Orbital_State.from_dict(
                x,
                ephem=ephem,
                density_model=density_model,
                fast=fast,
            )
            for x in lst
        ]

    if label is not None:
        return reconstruct(d[label])

    if len(d) == 1:
        return reconstruct(next(iter(d.values())))

    print(f"[load_orbital_states] Done.")
    return {k: reconstruct(v) for k, v in d.items()}
