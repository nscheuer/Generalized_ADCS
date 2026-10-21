"""Shared model selection and resources for an orbital simulation."""

from __future__ import annotations

from .atmosphere import Atmosphere, SMADAtmosphere
from .ephemeris import DE442s, Ephemeris
from .magnetic_field import IGRF14, MagneticField
from .star_catalogue import StarCatalogue


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
        :param ephemeris: Planetary ephemeris; defaults to bundled DE442s.
        :type ephemeris: Ephemeris or None
        :param star_catalogue: Star catalogue; defaults to curated navigation stars.
        :type star_catalogue: StarCatalogue or None
        :return: None
        :rtype: None
        """
        self.magnetic_field = magnetic_field if magnetic_field is not None else IGRF14()
        self.atmosphere = atmosphere if atmosphere is not None else SMADAtmosphere()
        self.ephemeris = ephemeris if ephemeris is not None else DE442s()
        self._star_catalogue_explicit = star_catalogue is not None
        self.star_catalogue = star_catalogue if star_catalogue is not None else StarCatalogue()

    @property
    def magnetic_model(self) -> str:
        r"""Return the selected magnetic model's canonical name.

        :return: Name used by legacy string model selectors.
        :rtype: str
        """
        return self.magnetic_field.name
