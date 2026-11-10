"""Compare true body rates with biased, noisy gyroscope measurements."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))
import ADCS


np.random.seed(17)
inertia = np.array([
    [0.0333, 0.0060, 0.0],
    [0.0060, 0.0333, 0.0],
    [0.0, 0.0, 0.00667],
])
rate_noise = ADCS.Noise(std_noise=np.deg2rad(0.05))  # rad/s, 1 s sample
bias_deg_per_second = np.array([0.2, -0.15, 0.1])
gyros = [
    ADCS.Gyro(
        axis=axis,
        sample_time=1.0,
        noise=rate_noise,
        bias=ADCS.Bias(bias=np.deg2rad(bias), std_bias=0.0),
    )
    for axis, bias in zip(np.eye(3), bias_deg_per_second)
]
satellite = ADCS.Satellite(mass=4.0, J_0=inertia, sensors=gyros)
initial_state = ADCS.State(
    w=np.deg2rad([3.5, -3.5, 0.1]), q=np.array([1.0, 0.0, 0.0, 0.0]),
)
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=initial_state, satellite=satellite, os0=orbit, dt=1.0, tf=180.0,
)

run = results.runs[0]
time_s = np.asarray(run.time_s)
omega_deg_s = np.rad2deg(np.asarray([state.w for state in run.state_hist]))
gyro_deg_s = np.rad2deg(np.vstack(run.sensor_hist))
residual_deg_s = gyro_deg_s - omega_deg_s
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1]})
colors = ("tab:blue", "tab:orange", "tab:green")
for axis, color, name in zip(range(3), colors, "xyz"):
    axes[0].plot(time_s, omega_deg_s[:, axis], color=color,
                 label=f"True $\\omega_{name}$")
    axes[0].plot(time_s, gyro_deg_s[:, axis], color=color, linestyle="--",
                 linewidth=1, label=f"Gyro $\\omega_{name}$")
    axes[1].plot(time_s, residual_deg_s[:, axis], color=color,
                 label=f"{name}-axis")
    axes[1].axhline(bias_deg_per_second[axis], color=color,
                    linestyle="--", linewidth=1,
                    label=f"{name} bias ({bias_deg_per_second[axis]:g} deg/s)")
axes[0].set_ylabel("Angular rate [deg/s]")
axes[0].set_title("True body rate and gyroscope measurement")
axes[0].legend(ncol=3, fontsize=9)
axes[0].grid(alpha=0.3)
axes[1].axhline(0.0, color="black", linewidth=0.8)
axes[1].set_ylabel("Measurement − true [deg/s]")
axes[1].set_xlabel("Time [s]")
axes[1].set_title("Gyro error: residual bias plus white noise")
axes[1].legend(ncol=3, fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
plot_path = Path(__file__).resolve().parents[3] / "docs/source/_static/tutorials/attitude_sensors/gyroscope_rates.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
