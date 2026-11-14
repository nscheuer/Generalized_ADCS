"""Plot coarse Sun sensor measurements without an attitude estimate."""

import numpy as np

import ADCS


np.random.seed(17)
axes = np.vstack((np.eye(3), -np.eye(3)))
sun_sensors = [
    ADCS.SunSensor(axis=axis, efficiency=1.0, sample_time=10.0,
                   noise=ADCS.Noise(std_noise=0.03))
    for axis in axes
]
gyros = [
    ADCS.Gyro(axis=axis, sample_time=10.0,
              noise=ADCS.Noise(std_noise=0.001),
              bias=ADCS.Bias(bias=bias))
    for axis, bias in zip(np.eye(3), [0.002, -0.0015, 0.001])
]
magnetometers = [
    ADCS.MTM(axis=axis, sample_time=10.0,
             noise=ADCS.Noise(std_noise=2e-6))
    for axis in np.eye(3)
]
satellite = ADCS.Satellite(mass=4.0, J_0=np.eye(3) * 0.02,
                           sensors=sun_sensors + gyros + magnetometers,
                           disturbances=[])
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.array([0.01, -0.008, 0.012]),
                 q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, dt=10.0, tf=600.0,
)
labels = ["Sun +X", "Sun +Y", "Sun +Z", "Sun -X", "Sun -Y", "Sun -Z",
          "Gyro X", "Gyro Y", "Gyro Z", "Mag X", "Mag Y", "Mag Z"]
units = ["reading"] * 6 + ["rad/s"] * 3 + ["T"] * 3
ADCS.plot(
    results,
    *(ADCS.plots.SensorsPlotSingle(i, title=label, label=label, units=unit)
      for i, (label, unit) in enumerate(zip(labels, units))),
    layout=(4, 3), figsize=(12, 11), title="Raw sensor measurements",
)
