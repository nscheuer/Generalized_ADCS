__all__ = ["IlluminationPlot"]

import numpy as np

from ..subplot import Subplot


class IlluminationPlot(Subplot):
    r"""
    Binary visualization of spacecraft Sun illumination state.

    This class plots whether the spacecraft is sunlit or in eclipse as a function
    of time, using a step-style representation. The plot is intended to give a
    clear, high-level view of illumination periods based on orbital state data
    available in the simulation object.

    The subplot focuses on user-configurable appearance options such as colors
    and title, without requiring knowledge of how illumination is computed.

    :param time:
        Name of the simulation attribute containing the time vector in seconds.
    :type time:
        str

    :param title:
        Title displayed at the top of the plot.
    :type title:
        str

    :param color_sunlit:
        Color used to indicate sunlit intervals.
    :type color_sunlit:
        str

    :param color_eclipse:
        Color used to indicate eclipse intervals.
    :type color_eclipse:
        str

    """
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Sun Illumination",
        color_sunlit: str = "tab:orange",
        color_eclipse: str = "tab:blue",
    ):
        self.time = time
        self.title = title
        self.color_sunlit = color_sunlit
        self.color_eclipse = color_eclipse

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        illum = np.array([os.is_sunlit() for os in sim.os_hist], dtype=int)

        ax.step(t, illum, where="post", color=self.color_sunlit)
        ax.fill_between(t, 0, illum, step="post", color=self.color_sunlit, alpha=0.3)
        ax.fill_between(t, illum, 1, step="post", color=self.color_eclipse, alpha=0.15)

        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Eclipse", "Sunlit"])

        ax.set_xlabel("Time [s]")
        ax.set_title(self.title, loc="left", pad=10)
        ax.grid(True, which="both")
