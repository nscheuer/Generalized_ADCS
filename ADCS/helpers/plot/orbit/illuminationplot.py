__all__ = ["IlluminationPlot"]

import numpy as np

from ..subplot import Subplot


class IlluminationPlot(Subplot):
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
