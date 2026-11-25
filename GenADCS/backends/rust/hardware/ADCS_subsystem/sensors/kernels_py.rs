use pyo3::prelude::*;
use pyo3::types::PyModule;

use super::kernels as k;

fn arr<const N: usize>(py: Python<'_>, value: &PyAny) -> PyResult<[f64; N]> {
    let values: Vec<f64> = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?
        .call_method0("ravel")?
        .extract()?;
    values
        .try_into()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err(format!("expected {N} values")))
}
fn mat3(py: Python<'_>, value: &PyAny) -> PyResult<k::Mat3> {
    let rows: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?
        .call_method0("tolist")?
        .extract()?;
    if rows.len() != 3 || rows.iter().any(|r| r.len() != 3) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected a 3x3 matrix",
        ));
    }
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j])))
}
fn mat2(py: Python<'_>, value: &PyAny) -> PyResult<k::Mat2> {
    let rows: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value, "float64"))?
        .call_method0("tolist")?
        .extract()?;
    if rows.len() != 2 || rows.iter().any(|r| r.len() != 2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "expected a 2x2 matrix",
        ));
    }
    Ok(std::array::from_fn(|i| std::array::from_fn(|j| rows[i][j])))
}
fn rows(py: Python<'_>, value: &PyAny) -> PyResult<Vec<Vec<f64>>> {
    py.import("numpy")?
        .call_method1("asarray", (value, "float64"))?
        .call_method0("tolist")?
        .extract()
}
fn out(py: Python<'_>, value: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .into())
}
fn vec3opt(py: Python<'_>, x: Option<&PyAny>) -> PyResult<Option<[f64; 3]>> {
    x.map(|v| arr(py, v)).transpose()
}

#[pyfunction]
#[pyo3(signature=(vector,attitude,orientation,misalignment,bias,noise_cholesky,saturation,with_bias,with_noise,noise_draw=None))]
fn vector_measurement(
    py: Python<'_>,
    vector: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    bias: &PyAny,
    noise_cholesky: &PyAny,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::vector_measurement(
            arr(py, vector)?,
            arr(py, attitude)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            arr(py, bias)?,
            mat3(py, noise_cholesky)?,
            saturation,
            with_bias,
            with_noise,
            vec3opt(py, noise_draw)?,
        ),
    )
}

#[pyfunction]
#[pyo3(signature=(angular_velocity_body,orientation,misalignment,bias,noise_cholesky,saturation,with_bias,with_noise,noise_draw=None))]
fn rate_measurement(
    py: Python<'_>,
    angular_velocity_body: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    bias: &PyAny,
    noise_cholesky: &PyAny,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::rate_measurement(
            arr(py, angular_velocity_body)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            arr(py, bias)?,
            mat3(py, noise_cholesky)?,
            saturation,
            with_bias,
            with_noise,
            vec3opt(py, noise_draw)?,
        ),
    )
}

#[pyfunction]
#[pyo3(signature=(sun_inertial,attitude,orientation,misalignment,normal,sensitivity,bias,noise_std,saturation,with_bias,with_noise,noise_draw=None))]
fn scalar_sun_measurement(
    py: Python<'_>,
    sun_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    normal: &PyAny,
    sensitivity: f64,
    bias: f64,
    noise_std: f64,
    saturation: f64,
    with_bias: bool,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<f64> {
    Ok(k::scalar_sun_measurement(
        arr(py, sun_inertial)?,
        arr(py, attitude)?,
        arr(py, orientation)?,
        arr(py, misalignment)?,
        arr(py, normal)?,
        sensitivity,
        bias,
        noise_std,
        saturation,
        with_bias,
        with_noise,
        noise_draw.map(|x| x.extract()).transpose()?,
    ))
}

#[pyfunction]
#[pyo3(signature=(direction_inertial,attitude,orientation,misalignment,bias,noise_cholesky,with_bias,with_noise,noise_draw=None))]
fn fine_direction_measurement(
    py: Python<'_>,
    direction_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    bias: &PyAny,
    noise_cholesky: &PyAny,
    with_bias: bool,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::fine_direction_measurement(
            arr(py, direction_inertial)?,
            arr(py, attitude)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            arr(py, bias)?,
            mat2(py, noise_cholesky)?,
            with_bias,
            with_noise,
            noise_draw.map(|x| arr::<2>(py, x)).transpose()?,
        ),
    )
}

#[pyfunction]
#[pyo3(signature=(position_inertial,velocity_inertial,bias,noise_cholesky,with_bias,with_noise,noise_draw=None))]
fn gnss_measurement(
    py: Python<'_>,
    position_inertial: &PyAny,
    velocity_inertial: &PyAny,
    bias: &PyAny,
    noise_cholesky: &PyAny,
    with_bias: bool,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::gnss_measurement(
            arr(py, position_inertial)?,
            arr(py, velocity_inertial)?,
            py.import("numpy")?
                .call_method1("asarray", (bias, "float64"))?
                .call_method0("ravel")?
                .extract()?,
            rows(py, noise_cholesky)?,
            with_bias,
            with_noise,
            noise_draw.map(|x| x.extract()).transpose()?,
        ),
    )
}

#[pyfunction]
#[pyo3(signature=(attitude,orientation,misalignment,noise_cholesky,with_noise,noise_draw=None))]
fn star_tracker_measurement(
    py: Python<'_>,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    noise_cholesky: &PyAny,
    with_noise: bool,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::star_tracker_measurement(
            arr(py, attitude)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            mat3(py, noise_cholesky)?,
            with_noise,
            vec3opt(py, noise_draw)?,
        ),
    )
}

#[pyfunction]
#[pyo3(signature=(noise_cholesky,noise_draw=None))]
fn small_rotation_noise_quaternion_from_cholesky(
    py: Python<'_>,
    noise_cholesky: &PyAny,
    noise_draw: Option<&PyAny>,
) -> PyResult<PyObject> {
    out(
        py,
        k::small_rotation_noise_quaternion_from_cholesky(
            mat3(py, noise_cholesky)?,
            vec3opt(py, noise_draw)?,
        ),
    )
}

#[pyfunction]
fn star_tracker_residual(
    py: Python<'_>,
    measured_attitude: &PyAny,
    estimated_attitude: &PyAny,
) -> PyResult<PyObject> {
    out(
        py,
        k::star_tracker_residual(arr(py, measured_attitude)?, arr(py, estimated_attitude)?),
    )
}
#[pyfunction]
fn star_tracker_mrp_jacobian(py: Python<'_>, delta_mrp: &PyAny) -> PyResult<PyObject> {
    out(py, k::star_tracker_mrp_jacobian(arr(py, delta_mrp)?))
}
#[pyfunction]
fn star_tracker_information(
    py: Python<'_>,
    noise_covariance: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
) -> PyResult<PyObject> {
    out(
        py,
        k::star_tracker_information(
            mat3(py, noise_covariance)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
        ),
    )
}
#[pyfunction]
fn magnetometer_information(
    py: Python<'_>,
    field_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    noise_covariance: &PyAny,
    field_covariance: &PyAny,
) -> PyResult<PyObject> {
    out(
        py,
        k::magnetometer_information(
            arr(py, field_inertial)?,
            arr(py, attitude)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            mat3(py, noise_covariance)?,
            mat3(py, field_covariance)?,
        ),
    )
}
#[pyfunction]
fn coarse_sun_available(
    py: Python<'_>,
    sun_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    normal: &PyAny,
    fov_half_angle: f64,
    enabled: bool,
    sun_visible: bool,
) -> PyResult<bool> {
    Ok(k::coarse_sun_available(
        arr(py, sun_inertial)?,
        arr(py, attitude)?,
        arr(py, orientation)?,
        arr(py, misalignment)?,
        arr(py, normal)?,
        fov_half_angle,
        enabled,
        sun_visible,
    ))
}
#[pyfunction]
fn coarse_sun_information(
    py: Python<'_>,
    sun_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    normal: &PyAny,
    sensitivity: f64,
    noise_variance: f64,
) -> PyResult<PyObject> {
    out(
        py,
        k::coarse_sun_information(
            arr(py, sun_inertial)?,
            arr(py, attitude)?,
            arr(py, orientation)?,
            arr(py, misalignment)?,
            arr(py, normal)?,
            sensitivity,
            noise_variance,
        ),
    )
}
#[pyfunction]
fn fine_direction_available(
    py: Python<'_>,
    direction_inertial: &PyAny,
    attitude: &PyAny,
    orientation: &PyAny,
    misalignment: &PyAny,
    fov_half_angles: &PyAny,
    enabled: bool,
    visible: bool,
) -> PyResult<bool> {
    Ok(k::fine_direction_available(
        arr(py, direction_inertial)?,
        arr(py, attitude)?,
        arr(py, orientation)?,
        arr(py, misalignment)?,
        arr(py, fov_half_angles)?,
        enabled,
        visible,
    ))
}

pub fn register(module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(vector_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(rate_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(scalar_sun_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(fine_direction_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(gnss_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(star_tracker_measurement, module)?)?;
    module.add_function(wrap_pyfunction!(
        small_rotation_noise_quaternion_from_cholesky,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(star_tracker_residual, module)?)?;
    module.add_function(wrap_pyfunction!(star_tracker_mrp_jacobian, module)?)?;
    module.add_function(wrap_pyfunction!(star_tracker_information, module)?)?;
    module.add_function(wrap_pyfunction!(magnetometer_information, module)?)?;
    module.add_function(wrap_pyfunction!(coarse_sun_available, module)?)?;
    module.add_function(wrap_pyfunction!(coarse_sun_information, module)?)?;
    module.add_function(wrap_pyfunction!(fine_direction_available, module)?)?;
    Ok(())
}
