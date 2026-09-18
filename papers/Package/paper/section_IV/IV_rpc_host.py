"""Run the Section IV RPC host simulation and save all plots.

The controller runs on the Raspberry Pi worker from :mod:`IV_rpc_worker`.
All generated figures, including the RPC timing diagnostic, are saved in
``section_IV/outputs`` before the plots are displayed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(REPO_ROOT))

import ADCS


OUTPUT_DIR = Path(__file__).with_name("outputs")


def save_plot(name: str) -> Path:
    """Save the current matplotlib figure in the Section IV output folder."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / name
    plt.gcf().savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {path}")
    return path


def _as_array(value) -> np.ndarray:
    if value is None:
        return np.empty(0, dtype=float)
    return np.asarray(value, dtype=float).reshape(-1)


def save_timing_plot(results: ADCS.SimulationResults) -> Path:
    """Save per-step RPC and local timing histories in milliseconds."""
    run = results.first()
    time_s = _as_array(getattr(run, "time_s", None))
    rtt = _as_array(getattr(run, "control_rpc_time_hist", None))
    server = _as_array(getattr(run, "control_rpc_server_time_hist", None))
    local = _as_array(getattr(run, "env_local_time_hist", None))
    dynamics = _as_array(getattr(run, "dynamics_time_hist", None))
    n = min([len(time_s), *[len(x) for x in (rtt, server, local, dynamics) if len(x)]]) if time_s.size else 0
    time_s = time_s[:n]

    fig, ax = plt.subplots(figsize=(10, 5))
    series = (
        ("RPC round trip", rtt),
        ("RPC server", server),
        ("RPC communication", np.maximum(rtt[:len(server)] - server, 0.0) if rtt.size and server.size else np.empty(0)),
        ("Local environment", local),
        ("Local dynamics", dynamics),
    )
    for label, values in series:
        if values.size:
            ax.plot(time_s[:len(values)], 1e3 * values[:n], label=label, alpha=0.85)
    ax.set(
        title="Section IV RPC timing",
        xlabel="Simulation time [s]",
        ylabel="Time per step [ms]",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return save_plot("iv_rpc_timing.png")


def main() -> ADCS.SimulationResults:
    np.random.seed(42)
    satellite = ADCS.satellite_factory.create_beavercube2_cubesat(estimated=False)
    x_0 = ADCS.State.from_array(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
    controller = ADCS.controller.MTQ_w_RW_LP(
        est_sat=satellite,
        p_gain=5e-5,
        d_gain=2e-3,
        c_gain=1e-3,
        h_target=np.zeros(3),
    )
    os0 = ADCS.Orbital_State(
        ephem=ADCS.Ephemeris(), J2000=0.22,
        R=7000 * np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
        V=np.array([8.0, 0.0, 0.0]),
    )
    goal = ADCS.goals.ECI_Goal(eci_vector=ADCS.helpers.normalize(np.ones(3)))
    remote_host = os.getenv("ADCS_REMOTE_HOST", "127.0.0.1")
    remote_port = int(os.getenv("ADCS_REMOTE_PORT", "5000"))

    start = time.perf_counter()
    results = ADCS.simulate_remote(
        x=x_0,
        satellite=satellite,
        os0=os0,
        controller=controller,
        goal=goal,
        dt=1.0,
        tf=1000.0,
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
    print(f"Host simulation wall time: {time.perf_counter() - start:.3f} s")

    ADCS.plot(
        results,
        ADCS.plots.AttitudePlot(sources=["real", "reference"]),
        layout=(1, 1),
        title="Section IV RPC: remote controller attitude",
    )
    save_plot("iv_rpc_attitude.png")

    ADCS.plot(
        results,
        ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
        ADCS.plots.ControlPlotCombined(title="Magnetorquer and wheel commands"),
        ADCS.plots.TargetHistogram(bin_width=5.0),
        ADCS.plots.TargetPlot(modes=["real_target"], title="Target tracking"),
        layout=(2, 2),
        title="Section IV RPC: remote controller results",
    )
    save_plot("iv_rpc_results.png")
    save_timing_plot(results)
    plt.show()
    return results


if __name__ == "__main__":
    main()
