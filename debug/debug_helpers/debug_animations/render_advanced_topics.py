"""Regenerate Advanced Topics GIFs with the Godot offline renderer.

Run from the repository root with the optional planner add-ons installed. This
editorial script is separate from the runnable tutorial examples.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import ADCS
from ADCS.helpers.plot.control.targetplot import TargetHistogram
from ADCS.helpers.math_helpers import angle_between_vectors, rot_mat

from ADCS.helpers.animations.godot.from_results import results_to_vizlog
from ADCS.helpers.animations.godot.assets.spacecraft import ensure_model_sidecar
from ADCS.helpers.animations.godot.render import CameraKey, RenderJob, render_video

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "docs/source/_static/tutorials/advanced_topics"
SCRIPTS = {
    "plan_and_track": ROOT / "examples/tutorials/advanced_topics/plan_and_track.py",
    "saltro": ROOT / "examples/tutorials/advanced_topics/saltro.py",
    "remote_execution": ROOT / "examples/tutorials/advanced_topics/remote_execution.py",
}


def simulation(name: str):
    """Run the example's setup, substituting a local controller for remote capture."""
    source = SCRIPTS[name].read_text()
    if name == "remote_execution":
        source = source.replace("ADCS.simulate_remote(", "ADCS.simulate(")
        source = source[:source.index("    remote=ADCS.remote.RemoteSimulationConfig(")] + ")\n"
    else:
        source = source[:source.index("ADCS.plot(")]
    namespace = {"__file__": str(SCRIPTS[name])}
    # ``ADCS.simulate`` displays the planned open-loop plot before it starts
    # closed-loop tracking. Keep that window non-blocking here so this
    # headless artifact generator can capture it along with the second plot.
    show = plt.show
    plt.show = lambda *args, **kwargs: None
    try:
        exec(compile(source, str(SCRIPTS[name]), "exec"), namespace)
    finally:
        plt.show = show
    return namespace["results"], namespace.get("controller")


def render(name: str):
    results, controller = simulation(name)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if name == "saltro":
        run = results.first()
        final_error = TargetHistogram()._get_final_error(run)
        final_rate = float(np.linalg.norm(run.state_hist[-1].w))
        wheel_commands = np.asarray(run.control_hist)[:, 3]
        planned_state = controller.active_trajectory.states[-1]
        planned_boresight = rot_mat(planned_state.q) @ run.satellite.get_boresight()
        open_error = float(np.rad2deg(angle_between_vectors(
            planned_boresight, np.array([0.0, 0.0, 1.0])
        )))
        print(f"SALTRO open-loop final error: {open_error:.3f} deg; "
              f"closed-loop final error: {final_error:.3f} deg; "
              f"final rate: {final_rate:.6f} rad/s; "
              f"peak wheel command: {np.max(np.abs(wheel_commands)):.6g}")
        if (open_error >= 1.0 or final_error >= 2.0 or final_rate >= 0.002
                or np.max(np.abs(wheel_commands)) <= 1e-7):
            raise RuntimeError("SALTRO simulation did not meet the tutorial performance check")
        plt.figure(plt.get_fignums()[-1]).savefig(OUTPUT / "saltro_planned.png", dpi=150)
        ADCS.plot(
            results,
            ADCS.plots.AngularVelocityPlotCombined(sources=["real"]),
            ADCS.plots.ControlPlotCombined(title="Magnetorquer Commands", units="Am²"),
            ADCS.plots.TargetHistogram(bin_width=5.0),
            ADCS.plots.TargetPlot(modes=["real_target"], title="Target Tracking"),
            layout=(2, 2), title="Closed-Loop Tracked Trajectory",
        )
        plt.figure(plt.get_fignums()[-1]).savefig(OUTPUT / "saltro_tracked.png", dpi=150)
        plt.close("all")
    with TemporaryDirectory(prefix=f"adcs-{name}-") as temp:
        work = Path(temp)
        model = ensure_model_sidecar("mirata") if name in ("plan_and_track", "saltro") else None
        if name in ("plan_and_track", "saltro") and model is None:
            raise FileNotFoundError("MiRaTa spacecraft model is unavailable")
        log = results_to_vizlog(results, work / "run.vizlog", model=model, torques="none")
        camera_distance = 2.5 if name == "remote_execution" else 12
        job = RenderJob(
            duration_s=6, fps=8, width=640, height=360, quality="fast",
            ui=False, night_vision=True,
            keys=[CameraKey(at=0, frame="inertial", dist_m=camera_distance,
                            yaw_deg=-135, pitch_deg=24)] if name == "saltro" else [
                CameraKey(at=0, dist_m=camera_distance, yaw_deg=-135, pitch_deg=24),
                CameraKey(at=6, dist_m=camera_distance, yaw_deg=-85, pitch_deg=24)],
        )
        render_video(log, job, work / "frames", codec="none")
        paths = sorted((work / "frames").glob("*.png"))
        if not paths:
            raise RuntimeError("Godot produced no frames")
        frames = [Image.open(p).convert("RGB") for p in paths]
        output = OUTPUT / f"{name}.gif"
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=125, loop=0, optimize=True)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=(*SCRIPTS, "all"))
    args = parser.parse_args()
    for selected in (SCRIPTS if args.name == "all" else (args.name,)):
        render(selected)
