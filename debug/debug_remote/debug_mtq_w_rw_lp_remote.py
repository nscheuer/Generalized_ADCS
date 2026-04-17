import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import ADCS as ADCS


def _build_problem():
    np.random.seed(42)
    real_sat = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    x_0 = np.array([0.0, 0.0, 0.0] + [1, 0, 0, 0] + [0.0])

    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=real_sat,
        p_gain=0.00005,
        d_gain=0.002,
        c_gain=0.001,
        h_target=np.array([0.0, 0.0, 0.0]),
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(),
        J2000=0.22,
        R=7000 * np.array([0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8, 0, 0]),
    )
    goal = ADCS.goals.ECI_Goal(eci_vector=ADCS.helpers.normalize(np.array([1.0, 1.0, 1.0])))
    return real_sat, controller, x_0, os0, goal


def run_remote_controller_demo(tf: float = 1000.0, dt: float = 1.0, show_plots: bool = True):
    real_sat, controller, x_0, os0, goal = _build_problem()

    remote_host = os.getenv("ADCS_REMOTE_HOST", "10.77.0.4")
    remote_port = int(os.getenv("ADCS_REMOTE_PORT", "5000"))
    print(f"[remote-demo] controller endpoint: {remote_host}:{remote_port}")
    print(
        "[remote-demo] if this fails with connection refused, start the server on the target machine with: "
        "ADCS_REMOTE_BIND_HOST=0.0.0.0 ADCS_REMOTE_PORT=5000 "
        "python debug/debug_remote/run_remote_mtq_w_rw_lp_server.py"
    )

    results = ADCS.simulate_remote(
        x=x_0,
        satellite=real_sat,
        os0=os0,
        controller=controller,
        goal=goal,
        dt=dt,
        tf=tf,
        remote=ADCS.remote.RemoteSimulationConfig(
            controller=ADCS.remote.ComponentLocation.REMOTE,
            estimator=ADCS.remote.ComponentLocation.LOCAL,
            orbit_estimator=ADCS.remote.ComponentLocation.LOCAL,
            host=remote_host,
            port=remote_port,
            timeout_s=0.5,
            retries=2,
        ),
    )

    run = results.first()
    rpc_times = np.asarray(run.control_rpc_time_hist, dtype=float)
    server_times = np.asarray(run.control_rpc_server_time_hist, dtype=float)

    print("Remote controller timing statistics")
    print(f"  host: {remote_host}:{remote_port}")
    print(f"  steps: {rpc_times.size}")
    print(f"  round-trip mean [s]: {np.nanmean(rpc_times):.6f}")
    print(f"  round-trip std  [s]: {np.nanstd(rpc_times):.6f}")
    print(f"  round-trip max  [s]: {np.nanmax(rpc_times):.6f}")
    print(f"  server compute mean [s]: {np.nanmean(server_times):.6f}")
    print(f"  server compute std  [s]: {np.nanstd(server_times):.6f}")
    print(f"  server compute max  [s]: {np.nanmax(server_times):.6f}")

    if show_plots:
        ADCS.plot(
            results,
            ADCS.plots.AttitudePlot(sources=["real", "reference"]),
            layout=(1, 1),
            title="Remote Controller: 3+1 LP Reduced",
        )

        ADCS.plot(
            results,
            ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
            ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
            ADCS.plots.TargetHistogram(bin_width=5.0),
            ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
            layout=(2, 2),
            title="Remote Controller: 3+1 LP Reduced",
        )

        plt.show()

    return results


if __name__ == "__main__":
    run_remote_controller_demo(tf=1000.0, dt=1.0)