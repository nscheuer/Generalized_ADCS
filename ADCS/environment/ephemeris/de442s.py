"""JPL's compact DE442s planetary ephemeris."""

from .de442 import DE442


class DE442s(DE442):
    """Use the optional DE442s kernel for years 1849 through 2150.

    If the kernel is missing, the parent loader can fetch it from the
    official JPL NAIF archive into the user cache.
    ``position_eci`` and ``sun_position_eci`` match the other DE models.
    """

    KERNEL_FILENAME = "de442s.bsp"
    ENV_PATH_VARIABLE = "ADCS_DE442S_EPHEMERIS_PATH"
    DOWNLOAD_URL = (
        "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
        "spk/planets/de442s.bsp"
    )


DE442S = DE442s

__all__ = ["DE442s", "DE442S"]
