"""Shared model selection and resources for an orbital simulation."""

from __future__ import annotations

import importlib
import json
import os
import threading
from datetime import datetime

from .atmosphere import Atmosphere, NRLMSISE00, SMADAtmosphere
from .ephemeris import DE421, DE442, DE442s, DE441, VSOP2013, Ephemeris
from .magnetic_field import CHAOS8, IGRF14, IGRF14Numba, IGRF14Ppigrf, WMM2025, MagneticField
from .star_catalogue import BrightStarCatalog, HipparcosCatalog, StarCatalogue

_DEFAULT_ENVIRONMENT = None
_DEFAULT_ENVIRONMENT_LOCK = threading.Lock()


def default_environment() -> "Environment":
    """Return the lazily created process-wide default model set."""
    global _DEFAULT_ENVIRONMENT
    if _DEFAULT_ENVIRONMENT is None:
        with _DEFAULT_ENVIRONMENT_LOCK:
            if _DEFAULT_ENVIRONMENT is None:
                _DEFAULT_ENVIRONMENT = Environment()
    return _DEFAULT_ENVIRONMENT


def _reset_default_environment_after_fork() -> None:
    """Ensure forked workers create their own environment and kernel handles."""
    global _DEFAULT_ENVIRONMENT, _DEFAULT_ENVIRONMENT_LOCK
    _DEFAULT_ENVIRONMENT = None
    _DEFAULT_ENVIRONMENT_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_default_environment_after_fork)


class Environment:
    """Models used to evaluate one orbit's environmental quantities.

    The orbit samples these models; :class:`Orbital_State` carries the sampled
    values to sensors, actuators, disturbances, and the attitude integrator.
    Custom atmosphere and ephemeris implementations can also be supplied here.
    """

    MagneticField = MagneticField
    Atmosphere = Atmosphere
    Ephemeris = Ephemeris
    StarCatalogue = StarCatalogue

    def __init__(
        self,
        magnetic_field: MagneticField | None = None,
        atmosphere: Atmosphere | None = None,
        ephemeris: Ephemeris | None = None,
        star_catalogue: StarCatalogue | None = None,
    ) -> None:
        r"""Select models used by orbital states and orbit propagation.

        :param magnetic_field: Field model; defaults to Numba IGRF-14.
        :type magnetic_field: MagneticField or None
        :param atmosphere: Density model; defaults to SMAD.
        :type atmosphere: Atmosphere or None
        :param ephemeris: Planetary ephemeris; defaults to DE421, downloaded on first use.
        :type ephemeris: Ephemeris or None
        :param star_catalogue: Star catalogue; defaults to curated navigation stars.
        :type star_catalogue: StarCatalogue or None
        :return: None
        :rtype: None
        """
        self.magnetic_field = magnetic_field if magnetic_field is not None else IGRF14()
        self.atmosphere = atmosphere if atmosphere is not None else SMADAtmosphere()
        self.ephemeris = ephemeris if ephemeris is not None else DE421()
        self._star_catalogue_explicit = star_catalogue is not None
        self.star_catalogue = star_catalogue if star_catalogue is not None else StarCatalogue()

    def to_json(self) -> str:
        """Serialize model names and JSON-safe settings for worker processes.

        Custom models must implement ``to_json_config()`` and a classmethod
        ``from_json_config(config)``. No model instances or binary handles cross
        the process boundary.
        """
        payload = {
            "version": 1,
            "magnetic_field": self._model_config(self.magnetic_field, "magnetic_field"),
            "atmosphere": self._model_config(self.atmosphere, "atmosphere"),
            "ephemeris": self._model_config(self.ephemeris, "ephemeris"),
            "star_catalogue": self._model_config(self.star_catalogue, "star_catalogue"),
            "star_catalogue_explicit": self._star_catalogue_explicit,
        }
        return json.dumps(payload)

    @staticmethod
    def _model_config(model, role):
        cls = type(model)
        if role == "atmosphere" and cls is SMADAtmosphere:
            return {"kind": "builtin", "name": "SMADAtmosphere", "config": {
                "altitude_range": model.altitude_range.tolist(), "rho_range": model.rho_range.tolist()}}
        if role == "atmosphere" and cls is NRLMSISE00:
            return {"kind": "builtin", "name": "NRLMSISE00", "config": {
                "version": model.version, "f107": model.f107, "f107a": model.f107a,
                "ap": model.ap, "reference_datetime": model.reference_datetime.isoformat(),
                "reference_latitude_deg": model.reference_latitude_deg,
                "reference_longitude_deg": model.reference_longitude_deg}}
        if role == "magnetic_field" and cls in (IGRF14, IGRF14Numba, IGRF14Ppigrf):
            return {"kind": "builtin", "name": "IGRF14", "config": {"implementation": model.name}}
        if role == "magnetic_field" and cls is WMM2025:
            return {"kind": "builtin", "name": "WMM2025", "config": {
                "kernel_path": str(model._custom_kernel_path) if model._custom_kernel_path else None}}
        if role == "magnetic_field" and cls is CHAOS8:
            return {"kind": "builtin", "name": "CHAOS8", "config": {
                "core_path": str(model.core_path), "static_path": str(model.static_path),
                "include_static": model.include_static}}
        if role == "ephemeris" and cls in (DE421, DE442, DE442s):
            path = getattr(model.planets, "path", None)
            return {"kind": "builtin", "name": cls.__name__, "config": {"filepath": str(path) if path else None}}
        if role == "ephemeris" and cls is DE441:
            paths = getattr(model.planets, "paths", ())
            return {"kind": "builtin", "name": "DE441", "config": {"filepath": str(paths[0]) if paths else None}}
        if role == "ephemeris" and cls is VSOP2013:
            return {"kind": "builtin", "name": "VSOP2013", "config": {
                "kernel_dir": str(model.kernel_dir) if model.kernel_dir else None}}
        if role == "star_catalogue" and cls is StarCatalogue:
            return {"kind": "builtin", "name": "StarCatalogue", "config": {}}
        if role == "star_catalogue" and cls is BrightStarCatalog:
            return {"kind": "builtin", "name": "BrightStarCatalog", "config": {
                "max_vmag": model.max_vmag, "kernel_path": str(model.kernel_path)}}
        if role == "star_catalogue" and cls is HipparcosCatalog:
            return {"kind": "builtin", "name": "HipparcosCatalog", "config": {
                "max_vmag": model.max_vmag, "kernel_dir": str(model.kernel_dir)}}

        to_config = getattr(model, "to_json_config", None)
        from_config = getattr(cls, "from_json_config", None)
        if (not callable(to_config) or not callable(from_config)
                or "<locals>" in cls.__qualname__):
            raise TypeError(
                f"Custom environment {role} {cls.__module__}.{cls.__qualname__} must implement "
                "importable to_json_config() and classmethod from_json_config(config) to run in Monte Carlo workers"
            )
        config = to_config()
        try:
            json.dumps(config)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{role} to_json_config() must return JSON-serializable data") from exc
        return {"kind": "custom", "class": f"{cls.__module__}:{cls.__qualname__}", "config": config}

    @classmethod
    def from_json(cls, serialized: str) -> "Environment":
        """Rebuild an environment from :meth:`to_json` in a worker process."""
        payload = json.loads(serialized)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported Environment JSON version: {payload.get('version')!r}")

        def restore(item):
            if item["kind"] == "custom":
                module_name, qualname = item["class"].split(":", 1)
                model_type = importlib.import_module(module_name)
                for part in qualname.split("."):
                    model_type = getattr(model_type, part)
                factory = getattr(model_type, "from_json_config", None)
                if not callable(factory):
                    raise TypeError(f"Custom environment model {item['class']} must implement from_json_config(config)")
                return factory(item["config"])
            if item["kind"] == "builtin":
                name, config = item["name"], item.get("config", {})
                models = {
                    "SMADAtmosphere": lambda: SMADAtmosphere(config["altitude_range"], config["rho_range"]),
                    "NRLMSISE00": lambda: NRLMSISE00(
                        version=config["version"], f107=config["f107"], f107a=config["f107a"], ap=config["ap"],
                        reference_datetime=datetime.fromisoformat(config["reference_datetime"]),
                        reference_latitude_deg=config["reference_latitude_deg"],
                        reference_longitude_deg=config["reference_longitude_deg"]),
                    "IGRF14": lambda: IGRF14(config["implementation"]),
                    "WMM2025": lambda: WMM2025(kernel_path=config["kernel_path"]),
                    "CHAOS8": lambda: CHAOS8(core_path=config["core_path"], static_path=config["static_path"],
                                              include_static=config["include_static"]),
                    "DE421": lambda: DE421(filepath=config["filepath"]) if config.get("filepath") else DE421(),
                    "DE442": lambda: DE442(filepath=config["filepath"]) if config.get("filepath") else DE442(),
                    "DE442s": lambda: DE442s(filepath=config["filepath"]) if config.get("filepath") else DE442s(),
                    "DE441": lambda: DE441(filepath=config["filepath"]) if config.get("filepath") else DE441(),
                    "VSOP2013": lambda: VSOP2013(kernel_dir=config["kernel_dir"]),
                    "StarCatalogue": lambda: StarCatalogue(),
                    "BrightStarCatalog": lambda: BrightStarCatalog(**config),
                    "HipparcosCatalog": lambda: HipparcosCatalog(**config),
                }
                factory = models.get(name)
                if factory is None:
                    raise ValueError(f"Unknown built-in environment model: {name!r}")
                return factory()
            raise ValueError(f"Unknown serialized environment component: {item['kind']!r}")

        environment = cls(
            magnetic_field=restore(payload["magnetic_field"]),
            atmosphere=restore(payload["atmosphere"]),
            ephemeris=restore(payload["ephemeris"]),
            star_catalogue=restore(payload["star_catalogue"]),
        )
        environment._star_catalogue_explicit = bool(payload.get("star_catalogue_explicit", False))
        return environment

    @property
    def magnetic_model(self) -> str:
        r"""Return the selected magnetic model's canonical name.

        :return: Name used by legacy string model selectors.
        :rtype: str
        """
        return self.magnetic_field.name
