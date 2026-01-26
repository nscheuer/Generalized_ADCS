import numpy as np

from ..subplot import Subplot


class EstimatorAlignmentPlot(Subplot):
    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Estimator Attitude Alignment Error",
        units: str = "deg",
    ):
        self.time = time
        self.title = title
        self.units = units

    def plot(self, ax, sim) -> None:
        # --- Validate required data ---
        if (
            sim.state_hist is None
            or sim.est_state_hist is None
        ):
            self._plot_no_data(ax)
            return

        X = np.asarray(sim.state_hist)
        Xhat = np.asarray(sim.est_state_hist)

        N = min(len(X), len(Xhat))
        if N == 0:
            self._plot_no_data(ax)
            return

        # --- Time axis ---
        t = getattr(sim, self.time, None)
        if t is not None:
            t = np.asarray(t)[:N]

        # --- Extract quaternions ---
        q_true = X[:N, 3:7]
        q_est = Xhat[:N, 3:7]

        # Normalize (safety)
        q_true = q_true / np.linalg.norm(q_true, axis=1, keepdims=True)
        q_est = q_est / np.linalg.norm(q_est, axis=1, keepdims=True)

        # --- Quaternion alignment error ---
        # |dot| handles q ↔ -q equivalence
        dots = np.abs(np.sum(q_true * q_est, axis=1))
        dots = np.clip(dots, -1.0, 1.0)

        error_rad = 2.0 * np.arccos(dots)

        if self.units == "deg":
            error = np.rad2deg(error_rad)
        else:
            error = error_rad

        # --- Plot ---
        if t is not None:
            ax.plot(t, error, label="Attitude Error")
            ax.set_xlabel("Time [s]")
        else:
            ax.plot(error, label="Attitude Error")
            ax.set_xlabel("Sample")

        ax.set_ylabel(f"Alignment Error [{self.units}]")
        ax.set_title(self.title)
        ax.grid(True, which="both")
        ax.legend()

    def _plot_no_data(self, ax):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(
            0.5,
            0.5,
            "No state / estimated state data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
