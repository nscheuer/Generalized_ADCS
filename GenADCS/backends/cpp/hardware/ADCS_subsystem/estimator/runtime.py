"""Convert sensor configuration once, then execute the native estimator loop."""
import numpy as np
from .attitude_estimator import AttitudeMEKF, Sensor, sample


def sample_estimator(times, vectors, rows, adcs, interval, format):
    if not isinstance(adcs.estimator, AttitudeMEKF):
        raise TypeError(f"unsupported estimator: {type(adcs.estimator).__name__}")
    fields = {name for name in (
        'estimated_attitude', 'estimated_rate', 'estimated_covariance',
        'estimated_attitude_covariance', 'estimated_rate_covariance',
        'measurement_innovation', 'measurement_innovation_covariance',
        'normalized_innovation_squared', 'normalized_estimation_error_squared',
        'measurement_residual', 'kalman_gain') if getattr(format, name, False)}
    devices, blocks, offsets = [], [], {}
    total = 0
    for sensor in adcs.sensors:
        name = type(sensor).__name__
        if name not in ('Gyroscope', 'AutonomousStarTracker'):
            continue
        field, width, code = (('gyroscope_reading', 3, 0) if name == 'Gyroscope'
                              else ('autonomous_star_tracker_reading', 4, 2))
        offset = offsets.get(field, 0)
        offsets[field] = offset + width
        native = Sensor()
        native.code = code
        native.orientation = sensor.orientation
        native.misalignment = sensor.misalignment
        native.bias = sensor.bias
        native.noise_cholesky = sensor.noise_cholesky
        native.enabled = sensor.enabled
        native.offset = total
        total += width
        devices.append(native)
        blocks.append(rows[field][:, offset:offset + width])
    readings = np.concatenate(blocks, axis=1) if blocks else np.empty((len(times), 0))
    rows.update(sample(adcs.estimator, times, vectors, readings, devices, interval, fields))
