//! Batched native sensor evaluation and PyO3 helpers for in-loop SimulationSampler.
use super::{
    sensor_kernels as k,
    simulation_sampling::{
        EstimateState, SensorSpec, SimulationSampler, SENSOR_CLASSES, SENSOR_NAMES, SENSOR_WIDTH,
    },
    Runtime,
};
use numpy::{ndarray::Array2, IntoPyArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(super) fn build_sampler_and_runtime(
    py: Python<'_>,
    epoch: f64,
    runtime_py: &PyAny,
    sensors: &PyAny,
    environment: &PyAny,
    seed: u64,
    estimator: Option<&PyAny>,
    nees: bool,
    force_ephemeris: bool,
) -> PyResult<(SimulationSampler, Runtime)> {
    let mut counts = [0; 8];
    let mut specs = Vec::new();
    for item in sensors.iter()? {
        let s = item?;
        let name = s.get_type().name()?;
        let kind = SENSOR_CLASSES
            .iter()
            .position(|x| *x == name)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported sensor {name}")))?;
        let mut c = SensorSpec {
            kind,
            offset: counts[kind],
            enabled: s.getattr("enabled")?.extract()?,
            ori: [1., 0., 0., 0.],
            mis: [1., 0., 0., 0.],
            bias: [0.; 6],
            chol: [[0.; 6]; 6],
            normal: [0.; 3],
            fov: [0.; 2],
            saturation: f64::INFINITY,
            sensitivity: 0.,
            std: 0.,
        };
        counts[kind] += SENSOR_WIDTH[kind];
        if kind != 6 {
            c.ori = array(py, s, "orientation")?;
            c.mis = array(py, s, "misalignment")?;
        }
        if kind == 3 {
            c.normal = array(py, s, "normal")?;
            c.bias[0] = s.getattr("bias")?.extract()?;
            c.sensitivity = s.getattr("sensitivity")?.extract()?;
            c.std = s.getattr("noise_std")?.extract()?;
        } else {
            let l: Vec<Vec<f64>> = s
                .getattr("noise_cholesky")?
                .call_method0("tolist")?
                .extract()?;
            let width = if kind == 2 { 3 } else { SENSOR_WIDTH[kind] };
            if l.len() != width || l.iter().any(|r| r.len() != width) {
                return Err(PyValueError::new_err("invalid sensor noise matrix"));
            }
            for i in 0..width {
                for j in 0..width {
                    c.chol[i][j] = l[i][j];
                }
            }
            if kind != 2 {
                let b: Vec<f64> = s.getattr("bias")?.call_method0("tolist")?.extract()?;
                if b.len() != width {
                    return Err(PyValueError::new_err("invalid sensor bias"));
                }
                c.bias[..width].copy_from_slice(&b);
            }
        }
        if [0, 1, 3, 7].contains(&kind) {
            c.saturation = s.getattr("saturation_value")?.extract()?;
        }
        if kind == 2 || kind == 3 {
            c.fov[0] = s.getattr("fov_half_angle")?.extract()?;
        }
        if kind == 4 || kind == 5 {
            c.fov = array(py, s, "fov_half_angles")?;
        }
        specs.push(c);
    }
    let estimate = match estimator.filter(|e| !e.is_none()) {
        None => None,
        Some(e) => match e.get_type().name()? {
            "AttitudeMEKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                Some(EstimateState::Mekf(
                    super::base::AttitudeEstimator::new(
                        array(py, e, "quaternion")?,
                        array(py, e, "rate")?,
                        std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                        std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                        Vec::new(),
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeEKF" => {
                let p = array::<49>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                Some(EstimateState::Ekf(
                    super::attitude_ekf::AttitudeEKF::new(
                        array(py, e, "quaternion")?,
                        array(py, e, "rate")?,
                        std::array::from_fn(|i| std::array::from_fn(|j| p[7 * i + j])),
                        std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                        Vec::new(),
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeSequentialUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::Ukf(
                    super::attitude_sequential_ukf::AttitudeSequentialUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeBatchedUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::BatchedUkf(
                    super::attitude_batched_ukf::AttitudeBatchedUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeSequentialSRUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::Srukf(
                    super::attitude_sequential_srukf::AttitudeSequentialSRUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            _ => return Err(PyValueError::new_err("unsupported Rust estimator")),
        },
    };
    let mut r = Runtime::from_py(py, runtime_py)?;
    let sun_needed = counts[2] + counts[3] + counts[4] > 0;
    if sun_needed && environment.getattr("ephemeris")?.is_none() {
        return Err(PyValueError::new_err(
            "Sun and star tracker sensors require an ephemeris",
        ));
    }
    if (sun_needed || force_ephemeris) && r.ephemeris.is_none() {
        let path: String = runtime_py.getattr("ephemeris_path")?.extract()?;
        r.ephemeris = Some(
            ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                .map_err(PyValueError::new_err)?,
        );
    }
    if counts[1] > 0 && r.magnetic.is_none() {
        r.magnetic = Some(match runtime_py.getattr("magnetic_kind")?.extract::<i32>()? {
            1 => magnetic_native::NativeMagneticField::igrf14().map_err(PyValueError::new_err)?,
            2 => magnetic_native::NativeMagneticField::chaos8(
                runtime_py.getattr("magnetic_include_static")?.extract()?,
            )
            .map_err(PyValueError::new_err)?,
            _ => {
                return Err(PyValueError::new_err(
                    "magnetometer requires a magnetic model",
                ))
            }
        });
    }
    Ok((
        SimulationSampler::new(specs, counts, estimate, epoch, seed, nees),
        r,
    ))
}

pub(super) fn sampler_to_pydict(py: Python<'_>, sampler: &SimulationSampler) -> PyResult<PyObject> {
    let result = PyDict::new(py);
    let rows = sampler.recorded_rows;
    for k in 0..8 {
        let cols = sampler.counts[k];
        let clean = Array2::from_shape_vec((rows, cols), sampler.recorded_clean[k].clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let reading = Array2::from_shape_vec((rows, cols), sampler.recorded_reading[k].clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        result.set_item(format!("{}_clean", SENSOR_NAMES[k]), clean.into_pyarray(py))?;
        result.set_item(
            format!("{}_reading", SENSOR_NAMES[k]),
            reading.into_pyarray(py),
        )?;
    }
    if sampler.estimate.is_some() {
        let est_width = if sampler.is_ekf { 109 } else { 96 };
        let e = Array2::from_shape_vec((rows, est_width), sampler.recorded_estimates.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let slices: [(&str, usize, usize); 11] = if sampler.is_ekf {
            [
                ("estimated_attitude", 0, 4),
                ("estimated_rate", 4, 7),
                ("estimated_covariance", 7, 56),
                ("measurement_innovation", 56, 59),
                ("measurement_innovation_covariance", 59, 68),
                ("normalized_innovation_squared", 68, 69),
                ("normalized_estimation_error_squared", 69, 70),
                ("measurement_residual", 70, 73),
                ("kalman_gain", 73, 91),
                ("estimated_attitude_covariance", 91, 100),
                ("estimated_rate_covariance", 100, 109),
            ]
        } else {
            [
                ("estimated_attitude", 0, 4),
                ("estimated_rate", 4, 7),
                ("estimated_covariance", 7, 43),
                ("measurement_innovation", 43, 46),
                ("measurement_innovation_covariance", 46, 55),
                ("normalized_innovation_squared", 55, 56),
                ("normalized_estimation_error_squared", 56, 57),
                ("measurement_residual", 57, 60),
                ("kalman_gain", 60, 78),
                ("estimated_attitude_covariance", 78, 87),
                ("estimated_rate_covariance", 87, 96),
            ]
        };
        for (name, start, end) in slices {
            result.set_item(
                name,
                e.slice(numpy::ndarray::s![.., start..end])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(result.into())
}
const NAMES: [&str; 8] = [
    "gyroscope",
    "magnetometer",
    "autonomous_star_tracker",
    "coarse_sun_sensor",
    "fine_sun_sensor",
    "fine_earth_sensor",
    "gnss",
    "accelerometer",
];
const CLASSES: [&str; 8] = [
    "Gyroscope",
    "Magnetometer",
    "AutonomousStarTracker",
    "CoarseSunSensor",
    "FineSunSensor",
    "FineEarthSensor",
    "GNSS",
    "Accelerometer",
];
const WIDTH: [usize; 8] = [3, 3, 4, 1, 2, 2, 6, 3];
struct Sensor {
    kind: usize,
    offset: usize,
    enabled: bool,
    ori: [f64; 4],
    mis: [f64; 4],
    bias: [f64; 6],
    chol: [[f64; 6]; 6],
    normal: [f64; 3],
    fov: [f64; 2],
    saturation: f64,
    sensitivity: f64,
    std: f64,
}
fn array<const N: usize>(py: Python<'_>, s: &PyAny, name: &str) -> PyResult<[f64; N]> {
    let a: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (s.getattr(name)?,))?
        .call_method0("ravel")?
        .call_method0("tolist")?
        .extract()?;
    a.try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must contain {N} values")))
}
#[pyfunction]
#[pyo3(signature=(times, states, epoch, interval, runtime, sensors, environment, seed=0, estimator=None, nees=false))]
pub fn sample_sensors(
    py: Python<'_>,
    times: numpy::PyReadonlyArray1<'_, f64>,
    states: numpy::PyReadonlyArray2<'_, f64>,
    epoch: f64,
    interval: f64,
    runtime: &PyAny,
    sensors: &PyAny,
    environment: &PyAny,
    seed: u64,
    estimator: Option<&PyAny>,
    nees: bool,
) -> PyResult<PyObject> {
    let times = times.as_array().to_vec();
    let states: Vec<Vec<f64>> = states
        .as_array()
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();
    if !epoch.is_finite()
        || times.iter().any(|x| !x.is_finite() || *x < 0.)
        || times.windows(2).any(|w| w[1] < w[0])
        || states.iter().flatten().any(|x| !x.is_finite())
    {
        return Err(PyValueError::new_err(
            "sensor trajectory must be finite and times nonnegative and ordered",
        ));
    }
    if !interval.is_finite()
        || interval <= 0.
        || times.len() != states.len()
        || states.iter().any(|x| x.len() < 13)
    {
        return Err(PyValueError::new_err(
            "invalid sensor trajectory or interval",
        ));
    }
    let mut counts = [0; 8];
    let mut config = Vec::new();
    for item in sensors.iter()? {
        let s = item?;
        let name = s.get_type().name()?;
        let kind = CLASSES
            .iter()
            .position(|x| *x == name)
            .ok_or_else(|| PyValueError::new_err(format!("unsupported sensor {name}")))?;
        let mut c = Sensor {
            kind,
            offset: counts[kind],
            enabled: s.getattr("enabled")?.extract()?,
            ori: [1., 0., 0., 0.],
            mis: [1., 0., 0., 0.],
            bias: [0.; 6],
            chol: [[0.; 6]; 6],
            normal: [0.; 3],
            fov: [0.; 2],
            saturation: f64::INFINITY,
            sensitivity: 0.,
            std: 0.,
        };
        counts[kind] += WIDTH[kind];
        if kind != 6 {
            c.ori = array(py, s, "orientation")?;
            c.mis = array(py, s, "misalignment")?;
        }
        if kind == 3 {
            c.normal = array(py, s, "normal")?;
            c.bias[0] = s.getattr("bias")?.extract()?;
            c.sensitivity = s.getattr("sensitivity")?.extract()?;
            c.std = s.getattr("noise_std")?.extract()?;
        } else {
            let l: Vec<Vec<f64>> = s
                .getattr("noise_cholesky")?
                .call_method0("tolist")?
                .extract()?;
            let width = if kind == 2 { 3 } else { WIDTH[kind] };
            if l.len() != width || l.iter().any(|r| r.len() != width) {
                return Err(PyValueError::new_err("invalid sensor noise matrix"));
            }
            for i in 0..width {
                for j in 0..width {
                    c.chol[i][j] = l[i][j];
                }
            }
            if kind != 2 {
                let b: Vec<f64> = s.getattr("bias")?.call_method0("tolist")?.extract()?;
                if b.len() != width {
                    return Err(PyValueError::new_err("invalid sensor bias"));
                }
                c.bias[..width].copy_from_slice(&b);
            }
        }
        if [0, 1, 3, 7].contains(&kind) {
            c.saturation = s.getattr("saturation_value")?.extract()?;
        }
        if kind == 2 || kind == 3 {
            c.fov[0] = s.getattr("fov_half_angle")?.extract()?;
        }
        if kind == 4 || kind == 5 {
            c.fov = array(py, s, "fov_half_angles")?;
        }
        config.push(c);
    }
    enum EstimateState {
        Mekf(super::base::AttitudeEstimator),
        Ekf(super::attitude_ekf::AttitudeEKF),
        Ukf(super::attitude_sequential_ukf::AttitudeSequentialUKF),
        BatchedUkf(super::attitude_batched_ukf::AttitudeBatchedUKF),
        Srukf(super::attitude_sequential_srukf::AttitudeSequentialSRUKF),
    }
    let mut estimate = match estimator.filter(|e| !e.is_none()) {
        None => None,
        Some(e) => match e.get_type().name()? {
            "AttitudeMEKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                Some(EstimateState::Mekf(
                    super::base::AttitudeEstimator::new(
                        array(py, e, "quaternion")?,
                        array(py, e, "rate")?,
                        std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                        std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                        Vec::new(),
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeEKF" => {
                let p = array::<49>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                Some(EstimateState::Ekf(
                    super::attitude_ekf::AttitudeEKF::new(
                        array(py, e, "quaternion")?,
                        array(py, e, "rate")?,
                        std::array::from_fn(|i| std::array::from_fn(|j| p[7 * i + j])),
                        std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                        Vec::new(),
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeSequentialUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::Ukf(
                    super::attitude_sequential_ukf::AttitudeSequentialUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeBatchedUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::BatchedUkf(
                    super::attitude_batched_ukf::AttitudeBatchedUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            "AttitudeSequentialSRUKF" => {
                let p = array::<36>(py, e, "covariance")?;
                let a = array::<9>(py, e, "acceleration_psd")?;
                let base = super::base::AttitudeEstimator::new(
                    array(py, e, "quaternion")?,
                    array(py, e, "rate")?,
                    std::array::from_fn(|i| std::array::from_fn(|j| p[6 * i + j])),
                    std::array::from_fn(|i| std::array::from_fn(|j| a[3 * i + j])),
                    Vec::new(),
                )
                .map_err(PyValueError::new_err)?;
                Some(EstimateState::Srukf(
                    super::attitude_sequential_srukf::AttitudeSequentialSRUKF::new(
                        base,
                        e.getattr("alpha")?.extract()?,
                        e.getattr("beta")?.extract()?,
                        e.getattr("kappa")?.extract()?,
                    )
                    .map_err(PyValueError::new_err)?,
                ))
            }
            _ => return Err(PyValueError::new_err("unsupported Rust estimator")),
        },
    };
    let is_ekf = matches!(estimate, Some(EstimateState::Ekf(_)));
    let mut estimator_config = super::attitude_mekf::SensorConfig::default();
    let mut total = 0;
    for s in &config {
        estimator_config.code.push(s.kind as i64);
        estimator_config.orientation.push(s.ori);
        estimator_config.misalignment.push(s.mis);
        estimator_config.bias.push(s.bias);
        estimator_config.noise_cholesky.push(s.chol);
        estimator_config.normal.push(s.normal);
        estimator_config.fov.push(s.fov);
        estimator_config
            .scalars
            .push([s.saturation, s.sensitivity, s.std]);
        estimator_config.reference.push([0.0; 3]);
        estimator_config.enabled.push(s.enabled);
        estimator_config.offsets.push(total);
        total += WIDTH[s.kind];
    }
    let mut estimates = estimate
        .as_ref()
        .map(|_| Array2::from_elem((times.len(), if is_ekf { 109 } else { 96 }), f64::NAN));
    let mut previous = 0.;
    let mut r = Runtime::from_py(py, runtime)?;
    let sun_needed = counts[2] + counts[3] + counts[4] > 0;
    if sun_needed {
        if environment.getattr("ephemeris")?.is_none() {
            return Err(PyValueError::new_err(
                "Sun and star tracker sensors require an ephemeris",
            ));
        }
        if r.ephemeris.is_none() {
            let path: String = runtime.getattr("ephemeris_path")?.extract()?;
            r.ephemeris = Some(
                ephemeris_native::NativeEphemeris::from_path(std::path::Path::new(&path))
                    .map_err(PyValueError::new_err)?,
            );
        }
    }
    if counts[1] > 0 && r.magnetic.is_none() {
        r.magnetic = Some(match runtime.getattr("magnetic_kind")?.extract::<i32>()? {
            1 => magnetic_native::NativeMagneticField::igrf14().map_err(PyValueError::new_err)?,
            2 => magnetic_native::NativeMagneticField::chaos8(
                runtime.getattr("magnetic_include_static")?.extract()?,
            )
            .map_err(PyValueError::new_err)?,
            _ => {
                return Err(PyValueError::new_err(
                    "magnetometer requires a magnetic model",
                ))
            }
        });
    }
    let (output, estimates) = py
        .allow_threads(move || -> Result<_, String> {
            let mut output: Vec<_> = counts
                .iter()
                .map(|&n| {
                    (
                        Array2::from_elem((times.len(), n), f64::NAN),
                        Array2::from_elem((times.len(), n), f64::NAN),
                    )
                })
                .collect();
            for (row, (&time, y)) in times.iter().zip(&states).enumerate() {
                let tick = (time / interval).round();
                if (time / interval - tick).abs() > 1e-9 {
                    continue;
                }
                let jd = epoch + time / 86400.;
                let q = [y[0], y[1], y[2], y[3]];
                let w = super::vec3(&y[4..]);
                let pos = super::vec3(&y[7..]);
                let vel = super::vec3(&y[10..]);
                let earth = pos.map(|x| -x);
                let mut sun = [0.; 3];
                let mut visible = false;
                if sun_needed {
                    let sp = super::scale3(
                        r.ephemeris
                            .as_ref()
                            .unwrap()
                            .position_km(
                                0,
                                magnetic_native::native_tdb_jd(jd).map_err(|e| e.to_string())?,
                            )
                            .map_err(|e| e.to_string())?,
                        1000.,
                    );
                    sun = super::sub3(sp, pos);
                    visible =
                        ephemeris_native::native_eclipse(pos, sp, r.radius, 696340000.).0 > 0.;
                }
                let field = if counts[1] > 0 {
                    r.magnetic
                        .as_ref()
                        .unwrap()
                        .field_inertial_tesla(pos, jd)
                        .map_err(|e| e.to_string())?
                } else {
                    [0.; 3]
                };
                let specific = if counts[7] > 0 {
                    let d = super::derivative_vec(y, jd, &r)?;
                    let g = gravity_native::native_acceleration(pos, r.mu, r.radius, &r.zonals)
                        .map_err(|e| e.to_string())?;
                    super::sub3(super::vec3(&d[10..]), g)
                } else {
                    [0.; 3]
                };
                for (index, s) in config.iter().enumerate() {
                    if !s.enabled {
                        continue;
                    }
                    let kind = s.kind;
                    let ori = s.ori;
                    let mis = s.mis;
                    if kind == 3 && !visible {
                        continue;
                    }
                    if kind == 4 || kind == 5 {
                        if !k::fine_direction_available(
                            if kind == 4 { sun } else { earth },
                            q,
                            ori,
                            mis,
                            s.fov,
                            true,
                            kind == 5 || visible,
                        ) {
                            continue;
                        }
                    }
                    if kind == 2
                        && [earth, sun].iter().any(|&v| {
                            let v = k::rotate_inertial_to_sensor(v, q, ori, mis);
                            v[2] > 0. && v[0].hypot(v[1]).atan2(v[2]) < s.fov[0]
                        })
                    {
                        continue;
                    }
                    let draw =
                        philox_native::philox::standard_normals(seed, tick as u64, index as u32, 6);
                    let l = std::array::from_fn(|i| std::array::from_fn(|j| s.chol[i][j]));
                    let b = [s.bias[0], s.bias[1], s.bias[2]];
                    let z = Some([draw[0], draw[1], draw[2]]);
                    for noisy in [false, true] {
                        let value: Vec<f64> = match kind {
                            0 => k::rate_measurement(
                                w,
                                ori,
                                mis,
                                b,
                                l,
                                s.saturation,
                                noisy,
                                noisy,
                                z,
                            )
                            .to_vec(),
                            1 | 7 => k::vector_measurement(
                                if kind == 1 { field } else { specific },
                                q,
                                ori,
                                mis,
                                b,
                                l,
                                s.saturation,
                                noisy,
                                noisy,
                                z,
                            )
                            .to_vec(),
                            2 => k::star_tracker_measurement(q, ori, mis, l, noisy, z).to_vec(),
                            3 => vec![k::scalar_sun_measurement(
                                sun,
                                q,
                                ori,
                                mis,
                                s.normal,
                                s.sensitivity,
                                s.bias[0],
                                s.std,
                                s.saturation,
                                noisy,
                                noisy,
                                Some(draw[0]),
                            )],
                            4 | 5 => k::fine_direction_measurement(
                                if kind == 4 { sun } else { earth },
                                q,
                                ori,
                                mis,
                                [s.bias[0], s.bias[1]],
                                [[s.chol[0][0], s.chol[0][1]], [s.chol[1][0], s.chol[1][1]]],
                                noisy,
                                noisy,
                                Some([draw[0], draw[1]]),
                            )
                            .to_vec(),
                            _ => (0..6)
                                .map(|i| {
                                    let base = if i < 3 { pos[i] } else { vel[i - 3] };
                                    if noisy {
                                        base + s.bias[i]
                                            + (0..6).map(|j| s.chol[i][j] * draw[j]).sum::<f64>()
                                    } else {
                                        base
                                    }
                                })
                                .collect(),
                        };
                        let dest = if noisy {
                            &mut output[kind].1
                        } else {
                            &mut output[kind].0
                        };
                        for (j, v) in value.into_iter().enumerate() {
                            dest[[row, s.offset + j]] = v;
                        }
                    }
                }
                if let Some(e) = estimate.as_mut() {
                    let mut readings = vec![f64::NAN; total];
                    for (i, s) in config.iter().enumerate() {
                        estimator_config.reference[i] = match s.kind {
                            1 => field,
                            3 | 4 => sun,
                            5 => earth,
                            _ => [0.0; 3],
                        };
                        for j in 0..WIDTH[s.kind] {
                            readings[estimator_config.offsets[i] + j] =
                                output[s.kind].1[[row, s.offset + j]];
                        }
                    }
                    let dt = time - previous;
                    previous = time;
                    match e {
                        EstimateState::Mekf(mekf) => {
                            let values = super::attitude_mekf::step_diagnostics(
                                mekf,
                                dt,
                                &readings,
                                &estimator_config,
                                y,
                                nees,
                            )
                            .map_err(str::to_owned)?;
                            for (j, v) in values.into_iter().enumerate() {
                                estimates.as_mut().unwrap()[[row, j]] = v;
                            }
                        }
                        EstimateState::Ekf(ekf) => {
                            let values = super::attitude_ekf::step_diagnostics(
                                ekf,
                                dt,
                                &readings,
                                &estimator_config,
                                y,
                                nees,
                            )
                            .map_err(str::to_owned)?;
                            for (j, v) in values.into_iter().enumerate() {
                                estimates.as_mut().unwrap()[[row, j]] = v;
                            }
                        }
                        EstimateState::Ukf(ukf) => {
                            let values = super::attitude_sequential_ukf::step_diagnostics(
                                ukf,
                                dt,
                                &readings,
                                &estimator_config,
                                y,
                                nees,
                            )
                            .map_err(str::to_owned)?;
                            for (j, v) in values.into_iter().enumerate() {
                                estimates.as_mut().unwrap()[[row, j]] = v;
                            }
                        }
                        EstimateState::BatchedUkf(batched_ukf) => {
                            let values = super::attitude_batched_ukf::step_diagnostics(
                                batched_ukf,
                                dt,
                                &readings,
                                &estimator_config,
                                y,
                                nees,
                            )
                            .map_err(str::to_owned)?;
                            for (j, v) in values.into_iter().enumerate() {
                                estimates.as_mut().unwrap()[[row, j]] = v;
                            }
                        }
                        EstimateState::Srukf(srukf) => {
                            let values = super::attitude_sequential_srukf::step_diagnostics(
                                srukf,
                                dt,
                                &readings,
                                &estimator_config,
                                y,
                                nees,
                            )
                            .map_err(str::to_owned)?;
                            for (j, v) in values.into_iter().enumerate() {
                                estimates.as_mut().unwrap()[[row, j]] = v;
                            }
                        }
                    }
                }
            }
            Ok((output, estimates))
        })
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    for (kind, (a, b)) in output.into_iter().enumerate() {
        result.set_item(format!("{}_clean", NAMES[kind]), a.into_pyarray(py))?;
        result.set_item(format!("{}_reading", NAMES[kind]), b.into_pyarray(py))?;
    }
    if let Some(e) = estimates {
        let slices: [(&str, usize, usize); 11] = if is_ekf {
            [
                ("estimated_attitude", 0, 4),
                ("estimated_rate", 4, 7),
                ("estimated_covariance", 7, 56),
                ("measurement_innovation", 56, 59),
                ("measurement_innovation_covariance", 59, 68),
                ("normalized_innovation_squared", 68, 69),
                ("normalized_estimation_error_squared", 69, 70),
                ("measurement_residual", 70, 73),
                ("kalman_gain", 73, 91),
                ("estimated_attitude_covariance", 91, 100),
                ("estimated_rate_covariance", 100, 109),
            ]
        } else {
            [
                ("estimated_attitude", 0, 4),
                ("estimated_rate", 4, 7),
                ("estimated_covariance", 7, 43),
                ("measurement_innovation", 43, 46),
                ("measurement_innovation_covariance", 46, 55),
                ("normalized_innovation_squared", 55, 56),
                ("normalized_estimation_error_squared", 56, 57),
                ("measurement_residual", 57, 60),
                ("kalman_gain", 60, 78),
                ("estimated_attitude_covariance", 78, 87),
                ("estimated_rate_covariance", 87, 96),
            ]
        };
        for (name, start, end) in slices {
            result.set_item(
                name,
                e.slice(numpy::ndarray::s![.., start..end])
                    .to_owned()
                    .into_pyarray(py),
            )?;
        }
    }
    Ok(result.into())
}
