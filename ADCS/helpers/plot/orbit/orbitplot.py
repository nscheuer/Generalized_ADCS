__all__ = ["OrbitPlot"]

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..subplot import Subplot
from ADCS.orbits.universal_constants import EarthConstants


def _normalize_orbit_sources(sources: list[str] | None) -> list[str]:
    if not sources:  # None or []
        return ["real"]
    allowed = {"real", "estimated"}
    out: list[str] = []
    for s in sources:
        s2 = str(s).strip().lower()
        if s2 not in allowed:
            raise ValueError(f"Invalid sources {s2!r}. Allowed: {sorted(allowed)}")
        if s2 not in out:
            out.append(s2)
    return out

def _get_R_series(sim, source: str) -> np.ndarray | None:
    """
    Return Nx3 ECI position history for a given source.

    - real      -> sim.os_hist[*].R
    - estimated -> sim.est_os_hist[*].os.R
    """
    if source == "real":
        hist = getattr(sim, "os_hist", None)
        get_R = lambda os: os.R
    elif source == "estimated":
        hist = getattr(sim, "est_os_hist", None)
        get_R = lambda os: os.os.R
    else:
        raise ValueError(f"Unknown source: {source}")

    if hist is None or len(hist) == 0:
        return None

    rows = [
        np.asarray(get_R(os), dtype=float).reshape(3)
        for os in hist
        if os is not None
    ]

    if not rows:
        return None

    return np.vstack(rows)


def _source_style_orbit3d(source: str) -> dict:
    return {
        "linestyle": "-" if source == "real" else "--"
    }


def _source_suffix_orbit3d(source: str) -> str:
    return " (real)" if source == "real" else " (est)"


class OrbitPlot(Subplot):
    def __init__(
        self,
        *,
        sources: list[str] | None = None,  # ["real", "estimated"]
        title: str = "Orbit (ECI)",
        orbit_colors: dict[str, str] | None = None,
        earth_color: str = "tab:blue",
        earth_alpha: float = 0.3,
        linewidth: float = 2.0,
    ):
        self.sources = _normalize_orbit_sources(sources)
        self.title = title
        self.orbit_colors = orbit_colors or {
            "real": "tab:red",
            "estimated": "tab:orange",
        }
        self.earth_color = earth_color
        self.earth_alpha = earth_alpha
        self.linewidth = linewidth

    def plot(self, ax, sim) -> None:
        fig = ax.figure
        fig.delaxes(ax)

        ax3d = fig.add_subplot(ax.get_subplotspec(), projection="3d")

        plotted_any = False
        all_R = []

        for source in self.sources:
            R = _get_R_series(sim, source)
            if R is None:
                continue

            style = _source_style_orbit3d(source)
            suf = _source_suffix_orbit3d(source)

            ax3d.plot(
                R[:, 0],
                R[:, 1],
                R[:, 2],
                color=self.orbit_colors.get(source, "k"),
                linewidth=self.linewidth,
                label="Orbit" + suf,
                **style,
            )

            all_R.append(R)
            plotted_any = True

        if not plotted_any:
            ax3d.set_title(self.title)
            ax3d.text2D(
                0.5,
                0.5,
                "No orbit history available",
                transform=ax3d.transAxes,
                ha="center",
                va="center",
            )
            return

        Re = EarthConstants.R_e

        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(-np.pi / 2, np.pi / 2, 25)
        u, v = np.meshgrid(u, v)

        x = Re * np.cos(v) * np.cos(u)
        y = Re * np.cos(v) * np.sin(u)
        z = Re * np.sin(v)

        ax3d.plot_surface(
            x,
            y,
            z,
            color=self.earth_color,
            alpha=self.earth_alpha,
            linewidth=0,
            antialiased=True,
        )

        R_all = np.vstack(all_R)
        max_range = np.max(np.linalg.norm(R_all, axis=1))
        lim = max_range * 1.1

        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)

        ax3d.set_box_aspect([1, 1, 1])

        ax3d.set_xlabel("X [km]")
        ax3d.set_ylabel("Y [km]")
        ax3d.set_zlabel("Z [km]")
        ax3d.set_title(self.title)

        ax3d.legend()
