import numpy as np
import matplotlib.gridspec as gridspec

from ..subplot import Subplot
from ADCS.helpers.math_helpers import rot_mat


def _normalize_modes(modes):
    if modes is None or len(modes) == 0:
        return ["real_target"]
    allowed = {"real_target", "est_target", "real_est", "directions3d"}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Invalid modes {bad}. Allowed: {sorted(allowed)}")
    # de-dup keep order
    out = []
    for m in modes:
        if m not in out:
            out.append(m)
    return out


def _quat_boresight_eci(q, bore_body):
    R_b2i = rot_mat(q)  # Body -> ECI
    v = R_b2i @ bore_body
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _angle_deg(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    u = u / nu
    v = v / nv
    dot = np.clip(float(np.dot(u, v)), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(dot)))


class TargetPlot(Subplot):
    """
    Three-way boresight / target / estimate comparison.

    modes:
      - "real_target": angle(real boresight, target)  [DEFAULT]
      - "est_target":  angle(estimated boresight, target)
      - "real_est":    angle(real boresight, estimated boresight)
      - "directions3d": one-sample 3D visualization of directions (real bore, est bore, target)

    Notes:
      - Uses sim.state_hist (real) quaternion at [3:7]
      - Uses sim.est_state_hist (estimated) quaternion at [3:7]
      - Uses sim.eci_target_hist (target direction in ECI)
      - Uses sim.satellite.boresight (body-fixed boresight)
    """

    def __init__(
        self,
        *,
        time: str = "time_s",
        title: str = "Target / Boresight Comparison",
        units: str = "deg",
        modes: list[str] | None = None,
        # for directions3d
        sample_index: int = -1,  # -1 means last sample
    ):
        self.time = time
        self.title = title
        self.units = units
        self.modes = _normalize_modes(modes)
        self.sample_index = sample_index

    def plot(self, ax, sim) -> None:
        # ---- required: real state, target, boresight ----
        if (
            sim.state_hist is None
            or sim.eci_target_hist is None
            or sim.satellite is None
            or not hasattr(sim.satellite, "boresight")
        ):
            self._plot_no_data(ax)
            return

        X_real = np.asarray(sim.state_hist)
        T = np.asarray(sim.eci_target_hist)

        N = min(len(X_real), len(T))
        if N <= 0:
            self._plot_no_data(ax)
            return

        X_est = None
        if sim.est_state_hist is not None and len(sim.est_state_hist) > 0:
            X_est = np.asarray(sim.est_state_hist)
            N = min(N, len(X_est))

        # time axis
        t = getattr(sim, self.time, None)
        if t is not None:
            t = np.asarray(t)[:N]

        # boresight in body
        bore_body = np.asarray(sim.satellite.boresight, dtype=float)
        nb = np.linalg.norm(bore_body)
        if nb == 0:
            self._plot_no_data(ax, msg="Satellite boresight has zero norm")
            return
        bore_body = bore_body / nb

        # compute requested series
        series = {}

        if "real_target" in self.modes:
            y = np.zeros(N)
            for i in range(N):
                q = X_real[i, 3:7]
                bore_eci = _quat_boresight_eci(q, bore_body)
                y[i] = _angle_deg(bore_eci, T[i])
            series["Real vs Target"] = y

        if "est_target" in self.modes:
            if X_est is None:
                series["Estimated vs Target (missing est_state_hist)"] = None
            else:
                y = np.zeros(N)
                for i in range(N):
                    qh = X_est[i, 3:7]
                    bore_eci_hat = _quat_boresight_eci(qh, bore_body)
                    y[i] = _angle_deg(bore_eci_hat, T[i])
                series["Estimated vs Target"] = y

        if "real_est" in self.modes:
            if X_est is None:
                series["Real vs Estimated (missing est_state_hist)"] = None
            else:
                y = np.zeros(N)
                for i in range(N):
                    q = X_real[i, 3:7]
                    qh = X_est[i, 3:7]
                    bore_eci = _quat_boresight_eci(q, bore_body)
                    bore_eci_hat = _quat_boresight_eci(qh, bore_body)
                    y[i] = _angle_deg(bore_eci, bore_eci_hat)
                series["Real vs Estimated"] = y

        want_3d = "directions3d" in self.modes

        # ---- layout ----
        n_series = sum(1 for v in series.values() if v is not None)
        if n_series == 0 and not want_3d:
            self._plot_no_data(ax, msg="No valid comparison modes available")
            return

        # remove the container axis and build sub-axes
        ax.set_frame_on(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)

        # rows: all time-series + optional 3D directions
        rows = n_series + (1 if want_3d else 0)
        gs = gridspec.GridSpecFromSubplotSpec(rows, 1, subplot_spec=ax.get_subplotspec(), hspace=0.35)

        r = 0
        # time-series axes
        for name, y in series.items():
            if y is None:
                continue
            ax_i = ax.figure.add_subplot(gs[r, 0])
            if t is not None:
                ax_i.plot(t, y, label=name)
                ax_i.set_xlabel("Time [s]")
            else:
                ax_i.plot(y, label=name)
                ax_i.set_xlabel("Sample")
            ax_i.set_ylabel(f"Error [{self.units}]")
            ax_i.grid(True, which="both")
            ax_i.legend()
            if r == 0:
                ax_i.set_title(self.title, loc="left", pad=10)
            r += 1

        # optional 3D “direction snapshot”
        if want_3d:
            ax3 = ax.figure.add_subplot(gs[r, 0], projection="3d")

            idx = self.sample_index
            if idx < 0:
                idx = N + idx  # python-style from end
            idx = int(np.clip(idx, 0, N - 1))

            q = X_real[idx, 3:7]
            bore_eci = _quat_boresight_eci(q, bore_body)
            target = np.asarray(T[idx], dtype=float)
            nt = np.linalg.norm(target)
            target = target / nt if nt > 0 else target

            bore_hat = None
            if X_est is not None:
                qh = X_est[idx, 3:7]
                bore_hat = _quat_boresight_eci(qh, bore_body)

            # draw unit vectors from origin
            def _vec(v):
                return np.array([0, v[0]]), np.array([0, v[1]]), np.array([0, v[2]])

            xs, ys, zs = _vec(bore_eci)
            ax3.plot(xs, ys, zs, label="Real boresight")

            xs, ys, zs = _vec(target)
            ax3.plot(xs, ys, zs, label="Target")

            if bore_hat is not None:
                xs, ys, zs = _vec(bore_hat)
                ax3.plot(xs, ys, zs, label="Estimated boresight")

            ax3.set_xlim(-1, 1)
            ax3.set_ylim(-1, 1)
            ax3.set_zlim(-1, 1)
            ax3.set_box_aspect([1, 1, 1])
            ax3.set_title(f"Direction snapshot (k={idx})")
            ax3.legend()

    def _plot_no_data(self, ax, msg="No target / state data available"):
        ax.axis("off")
        ax.set_title(self.title, loc="left", pad=10)
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
