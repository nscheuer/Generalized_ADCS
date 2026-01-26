import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot


class OrbitPositionPlot(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Orbit Position (ECI)",
        units: str = "km",
        colors=("tab:blue", "tab:orange", "tab:green"),
        mag_color: str = "tab:red",
        log_y: bool = False,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.mag_color = mag_color
        self.log_y = log_y

    def plot(self, ax, sim) -> None:
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        gs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=ax.get_subplotspec())

        ax_rx = ax.figure.add_subplot(gs[0, 0])
        ax_rmag = ax.figure.add_subplot(gs[0, 1])
        ax_ry = ax.figure.add_subplot(gs[1, 0])
        ax_rz = ax.figure.add_subplot(gs[1, 1])

        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax_text = ax.figure.add_subplot(ax.get_subplotspec())
            ax_text.axis("off")
            ax_text.set_title(self.title, loc="left", pad=10)
            ax_text.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        R = np.vstack([np.asarray(os.R) for os in sim.os_hist])
        rmag = np.linalg.norm(R, axis=1)

        labels = [r"$r_x$", r"$r_y$", r"$r_z$"]
        axes = [ax_rx, ax_ry, ax_rz]

        for i, ax_i in enumerate(axes):
            ax_i.plot(t, R[:, i], color=self.colors[i], label=labels[i])
            ax_i.set_ylabel(f"{labels[i]} [{self.units}]")
            if self.log_y:
                ax_i.set_yscale("log")
            ax_i.legend()
            ax_i.grid(True, which="both")

        ax_rmag.plot(t, rmag, color=self.mag_color, label=r"$\|r\|$")
        ax_rmag.set_ylabel(f"$\\|r\\|$ [{self.units}]")
        if self.log_y:
            ax_rmag.set_yscale("log")
        ax_rmag.legend()
        ax_rmag.grid(True, which="both")

        ax_ry.set_xlabel("Time [s]")
        ax_rz.set_xlabel("Time [s]")
        ax_rx.set_title(self.title, loc="left", pad=10)


class OrbitPositionPlotSingle(Subplot):
    def __init__(
        self,
        *,
        component: str,
        time: str = "time_s",
        title: str | None = None,
        units: str = "km",
        color: str | None = None,
        log_y: bool = False,
        labels: dict[str, str] | None = None,
    ):
        if component not in {"x", "y", "z", "m"}:
            raise ValueError("component must be one of 'x', 'y', 'z', or 'm'")
        self.component = component
        self.time = time
        self.units = units
        self.color = color
        self.log_y = log_y
        self.labels = labels or {
            "x": r"$r_x$",
            "y": r"$r_y$",
            "z": r"$r_z$",
            "m": r"$\|r\|$",
        }
        self.title = title

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title or "Orbit Position (ECI)", loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        R = np.vstack([np.asarray(os.R) for os in sim.os_hist])

        if self.component == "x":
            y = R[:, 0]
            label = self.labels["x"]
            color = self.color or "tab:blue"
            title = self.title or "Orbit Position $r_x$ (ECI)"
        elif self.component == "y":
            y = R[:, 1]
            label = self.labels["y"]
            color = self.color or "tab:orange"
            title = self.title or "Orbit Position $r_y$ (ECI)"
        elif self.component == "z":
            y = R[:, 2]
            label = self.labels["z"]
            color = self.color or "tab:green"
            title = self.title or "Orbit Position $r_z$ (ECI)"
        else:
            y = np.linalg.norm(R, axis=1)
            label = self.labels["m"]
            color = self.color or "tab:red"
            title = self.title or "Orbit Position $\\|r\\|$ (ECI)"

        ax.plot(t, y, color=color, label=label)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [{self.units}]")
        ax.set_title(title)
        if self.log_y:
            ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both")


class OrbitPositionPlotCombined(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Orbit Position (ECI)",
        units: str = "km",
        colors=("tab:blue", "tab:orange", "tab:green"),
        log_y: bool = False,
        labels: list[str] | None = None,
    ):
        self.time = time
        self.title = title
        self.units = units
        self.colors = colors
        self.log_y = log_y
        self.labels = labels or [r"$r_x$", r"$r_y$", r"$r_z$"]

    def plot(self, ax, sim) -> None:
        t = getattr(sim, self.time)

        if sim.os_hist is None or len(sim.os_hist) == 0:
            ax.axis("off")
            ax.set_title(self.title, loc="left", pad=10)
            ax.text(0.5, 0.5, "No os_hist available", ha="center", va="center")
            return

        R = np.vstack([np.asarray(os.R) for os in sim.os_hist])

        for i in range(3):
            ax.plot(
                t,
                R[:, i],
                color=self.colors[i],
                label=self.labels[i],
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Position [{self.units}]")
        ax.set_title(self.title, loc="left", pad=10)

        if self.log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True, which="both")
