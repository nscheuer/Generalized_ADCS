use pyo3::exceptions::PyValueError;
use pyo3::types::PyTuple;

const EARTH_ZONAL_COEFFICIENTS: [f64; 5] = [
    1.08262668e-3,
    -2.5324105e-6,
    -1.6198976e-6,
    -2.2729608e-7,
    5.406812e-7,
];
const EARTH_MU_M3_S2: f64 = 3.986004418e14;
const EARTH_REFERENCE_RADIUS_M: f64 = 6_378_137.0;

fn acceleration(position: [f64; 3], mu: f64, reference_radius: f64, coefficients: &[f64]) -> PyResult<[f64; 3]> {
    let [x, y, z] = position;
    let r2 = x * x + y * y + z * z;
    let r = r2.sqrt();
    if r == 0.0 {
        return Err(PyValueError::new_err("Gravity is undefined at the body's centre"));
    }
    let inv_r = 1.0 / r;
    let s = z * inv_r;
    let ratio = reference_radius * inv_r;
    let mut ratio_n = ratio * ratio;
    let p = [
        0.5 * (3.0 * s * s - 1.0),
        0.5 * (5.0 * s.powi(3) - 3.0 * s),
        (35.0 * s.powi(4) - 30.0 * s * s + 3.0) / 8.0,
        (63.0 * s.powi(5) - 70.0 * s.powi(3) + 15.0 * s) / 8.0,
        (231.0 * s.powi(6) - 315.0 * s.powi(4) + 105.0 * s * s - 5.0) / 16.0,
    ];
    let dp = [
        3.0 * s,
        0.5 * (15.0 * s * s - 3.0),
        (140.0 * s.powi(3) - 60.0 * s) / 8.0,
        (315.0 * s.powi(4) - 210.0 * s * s + 15.0) / 8.0,
        (1386.0 * s.powi(5) - 1260.0 * s.powi(3) + 210.0 * s) / 16.0,
    ];
    let mut radial = 1.0;
    let mut z_gradient = 0.0;
    for k in 0..coefficients.len().min(5) {
        let jn = coefficients[k];
        let n = (k + 2) as f64;
        radial -= (n + 1.0) * jn * ratio_n * p[k];
        z_gradient += jn * ratio_n * dp[k];
        ratio_n *= ratio;
    }
    let mu_r3 = mu / (r2 * r);
    let base = -mu_r3 * radial;
    Ok([
        base * x + mu_r3 * z_gradient * (-s * x),
        base * y + mu_r3 * z_gradient * (-s * y),
        base * z + mu_r3 * z_gradient * (r - s * z),
    ])
}

fn vector3(py: Python<'_>, value: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py.import("numpy")?.call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(format!("{name} must be a 3-vector")));
    }
    Ok([values[0], values[1], values[2]])
}

#[pyfunction]
fn gravity_acceleration_inertial_m_s2(
    py: Python<'_>,
    position_inertial_m: &PyAny,
    epoch_jd: f64,
    gravitational_parameter_m3_s2: f64,
    reference_radius_m: f64,
    zonal_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let position = vector3(py, position_inertial_m, "position_inertial_m")?;
    let coefficients: Vec<f64> = py.import("numpy")?.call_method1("asarray", (zonal_coefficients,))?
        .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    let _ = epoch_jd;
    Ok(acceleration(position, gravitational_parameter_m3_s2, reference_radius_m, &coefficients)?.to_pyarray(py).into_py(py))
}

#[pyclass(subclass)]
struct JGravityModel {
    degrees: Vec<i32>,
    zonal_coefficients: [f64; 5],
}

impl JGravityModel {
    fn configured(degrees: Vec<i32>, coefficients: Option<Vec<f64>>) -> PyResult<Self> {
        let mut degrees = degrees;
        degrees.sort_unstable();
        degrees.dedup();
        if degrees.iter().any(|degree| !(2..=6).contains(degree)) {
            return Err(PyValueError::new_err("J-gravity degrees must be selected from 2 through 6"));
        }
        if coefficients.as_ref().is_some_and(|values| values.len() < 5) {
            return Err(PyValueError::new_err("Provide five coefficients ordered as J2, J3, J4, J5, J6"));
        }
        let mut selected = [0.0; 5];
        let source = coefficients.unwrap_or_else(|| EARTH_ZONAL_COEFFICIENTS.to_vec());
        for degree in &degrees {
            selected[(degree - 2) as usize] = source[(degree - 2) as usize];
        }
        Ok(Self { degrees, zonal_coefficients: selected })
    }
}

#[pymethods]
impl JGravityModel {
    #[new]
    #[pyo3(signature = (degrees=None, coefficients=None))]
    fn new(degrees: Option<Vec<i32>>, coefficients: Option<Vec<f64>>) -> PyResult<Self> {
        Self::configured(degrees.unwrap_or_else(|| vec![2, 3, 4, 5, 6]), coefficients)
    }

    #[getter]
    fn degrees(&self, py: Python<'_>) -> PyObject {
        PyTuple::new(py, self.degrees.iter()).into()
    }

    #[getter]
    fn zonal_coefficients(&self, py: Python<'_>) -> PyObject {
        self.zonal_coefficients.to_pyarray(py).into_py(py)
    }

    #[classattr]
    fn degree() -> Option<i32> {
        None
    }

    #[pyo3(signature = (position_inertial_m, epoch_jd=0.0))]
    fn evaluate(&self, py: Python<'_>, position_inertial_m: &PyAny, epoch_jd: f64) -> PyResult<PyObject> {
        let position = vector3(py, position_inertial_m, "position_inertial_m")?;
        let _ = epoch_jd;
        Ok(acceleration(position, EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M, &self.zonal_coefficients)?.to_pyarray(py).into_py(py))
    }

    #[pyo3(signature = (position_inertial_m, epoch_jd=0.0))]
    fn acceleration(&self, py: Python<'_>, position_inertial_m: &PyAny, epoch_jd: f64) -> PyResult<PyObject> {
        self.evaluate(py, position_inertial_m, epoch_jd)
    }

    #[staticmethod]
    fn acceleration_inertial_m_s2(
        py: Python<'_>,
        position_inertial_m: &PyAny,
        epoch_jd: f64,
        gravitational_parameter_m3_s2: f64,
        reference_radius_m: f64,
        zonal_coefficients: &PyAny,
    ) -> PyResult<PyObject> {
        gravity_acceleration_inertial_m_s2(py, position_inertial_m, epoch_jd,
            gravitational_parameter_m3_s2, reference_radius_m, zonal_coefficients)
    }
}
