"""Determine attitude from the same Sun and magnetic sensors using TRIAD."""

import numpy as np

import ADCS
from ADCS.estimators.attitude_determination import triad
from ADCS.estimators.initialization import vector_observations


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
sensors = sun_sensors + gyros + magnetometers
satellite = ADCS.Satellite(mass=4.0, J_0=np.eye(3) * 0.02,
                           sensors=sensors, disturbances=[])
orbit = ADCS.Orbital_State(
    ephem=ADCS.Ephemeris(), J2000=0.22,
    R=np.array([7000.0, 0.0, 0.0]), V=np.array([0.0, 7.5, 0.0]),
)
results = ADCS.simulate(
    x=ADCS.State(w=np.array([0.01, -0.008, 0.012]),
                 q=np.array([1.0, 0.0, 0.0, 0.0])),
    satellite=satellite, os0=orbit, dt=10.0, tf=600.0,
)

# Keep the measurement layout identical to the simulated satellite.
est_satellite = ADCS.EstimatedSatellite(
    mass=4.0, J_0=np.eye(3) * 0.02, sensors=sensors, disturbances=[],
)
run = results.first()
valid_times, true_states, estimated_states = [], [], []
for time, truth, reading, orbital_state in zip(
    run.time_s, run.state_hist, run.sensor_hist, run.os_hist
):
    observations = {
        item.kind: item
        for item in vector_observations(est_satellite, reading, orbital_state)
    }
    if not {"sun", "magnetic_field"} <= observations.keys():
        continue
    sun, magnetic = observations["sun"], observations["magnetic_field"]
    if abs(np.dot(sun.reference, magnetic.reference)) > 0.99:
        continue  # TRIAD cannot resolve nearly parallel directions.
    attitude = triad(sun.body, magnetic.body,
                     sun.reference, magnetic.reference).quaternion
    valid_times.append(time)
    true_states.append(truth)
    estimated_states.append(ADCS.State(w=np.asarray(reading)[6:9], q=attitude))

if not valid_times:
    raise RuntimeError("No valid Sun and magnetic field direction pairs")
run.time_s = np.asarray(valid_times)
run.state_hist = true_states
run.est_state_hist = estimated_states
run.target_hist = np.asarray([state.q for state in true_states])
run.boresight_hist = np.tile([1.0, 0.0, 0.0], (len(valid_times), 1))
ADCS.plot(results, ADCS.plots.TargetPlot(modes=["est_target"],
                                           title="TRIAD attitude angle error"),
          layout=(1, 1), title="TRIAD attitude determination")
