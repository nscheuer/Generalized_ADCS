import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ..subplot import Subplot
from ADCS.orbits.universal_constants import EarthConstants


class OrbitPlot(Subplot):
    def __init__(
        self,
        *,
        title: str = "Orbit (ECI)",
        orbit_color: str = "tab:red",
        earth_color: str = "tab:blue",
        earth_alpha: float = 0.3,
        linewidth: float = 2.0,
    ):
        self.title = title
        self.orbit_color = orbit_color
        self.earth_color = earth_color
        self.earth_alpha = earth_alpha
        self.linewidth = linewidth

    def plot(self, ax, sim) -> None:
        fig = ax.figure
        fig.delaxes(ax)

        ax3d = fig.add_subplot(ax.get_subplotspec(), projection="3d")

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax3d.set_title(self.title)
            return

        R = np.vstack([np.asarray(os.R) for os in sim.os_hist])

        ax3d.plot(
            R[:, 0],
            R[:, 1],
            R[:, 2],
            color=self.orbit_color,
            linewidth=self.linewidth,
            label="Orbit",
        )

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

        max_range = np.max(np.linalg.norm(R, axis=1))
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
