//! PyO3 bindings and NumPy conversions for native frame conversions.

use super::*;
use pyo3::exceptions::PyValueError;

fn optional_fixed3(py: Python<'_>, value: Option<&PyAny>, name: &str) -> PyResult<Option<V>> {
    value.map(|value| vec3(py, value, name)).transpose()
}

fn optional_q(py: Python<'_>, value: Option<&PyAny>) -> PyResult<Option<[f64; 4]>> {
    value
        .map(|value| {
            let values: Vec<f64> = py
                .import("numpy")?
                .call_method1("asarray", (value,))?
                .call_method1("astype", ("float64",))?
                .call_method0("tolist")?
                .extract()?;
            let quaternion: [f64; 4] = values
                .try_into()
                .map_err(|_| PyValueError::new_err("q_body_from_gcrf has an invalid shape"))?;
            if quaternion.iter().any(|value| !value.is_finite()) {
                return Err(PyValueError::new_err(
                    "q_body_from_gcrf must contain only finite values",
                ));
            }
            Ok(quaternion)
        })
        .transpose()
}

#[pyfunction]
#[pyo3(signature=(source,target,*,jd_ut1=None,jd_tt=None,xp=0.0,yp=0.0,position_gcrf=None,velocity_gcrf=None,q_body_from_gcrf=None))]
pub(super) fn frame_rotation(
    py: Python<'_>,
    source: &str,
    target: &str,
    jd_ut1: Option<f64>,
    jd_tt: Option<f64>,
    xp: f64,
    yp: f64,
    position_gcrf: Option<&PyAny>,
    velocity_gcrf: Option<&PyAny>,
    q_body_from_gcrf: Option<&PyAny>,
) -> PyResult<PyObject> {
    let position = optional_fixed3(py, position_gcrf, "position_gcrf")?;
    let velocity = optional_fixed3(py, velocity_gcrf, "velocity_gcrf")?;
    let quaternion = optional_q(py, q_body_from_gcrf)?;
    let matrix = conversions::context_rotation(
        conversions::Frame::parse(source).map_err(PyValueError::new_err)?,
        conversions::Frame::parse(target).map_err(PyValueError::new_err)?,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position,
        velocity,
        quaternion,
    )
    .map_err(PyValueError::new_err)?;
    Ok(pymat(py, &matrix))
}

#[pyfunction]
#[pyo3(signature=(vector,source,target,*,jd_ut1=None,jd_tt=None,xp=0.0,yp=0.0,position_gcrf=None,velocity_gcrf=None,q_body_from_gcrf=None))]
pub(super) fn convert_vector(
    py: Python<'_>,
    vector: &PyAny,
    source: &str,
    target: &str,
    jd_ut1: Option<f64>,
    jd_tt: Option<f64>,
    xp: f64,
    yp: f64,
    position_gcrf: Option<&PyAny>,
    velocity_gcrf: Option<&PyAny>,
    q_body_from_gcrf: Option<&PyAny>,
) -> PyResult<PyObject> {
    let position = optional_fixed3(py, position_gcrf, "position_gcrf")?;
    let velocity = optional_fixed3(py, velocity_gcrf, "velocity_gcrf")?;
    let quaternion = optional_q(py, q_body_from_gcrf)?;
    let matrix = conversions::context_rotation(
        conversions::Frame::parse(source).map_err(PyValueError::new_err)?,
        conversions::Frame::parse(target).map_err(PyValueError::new_err)?,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position,
        velocity,
        quaternion,
    )
    .map_err(PyValueError::new_err)?;
    pyarr(py, mv(matrix, vec3(py, vector, "vector")?))
}

#[pyfunction]
pub(super) fn ecef_to_geodetic(py: Python<'_>, position_ecef_m: &PyAny) -> PyResult<PyObject> {
    let position = vec3(py, position_ecef_m, "position_ecef_m")?;
    pyarr(
        py,
        conversions::ecef_to_geodetic(position).map_err(PyValueError::new_err)?,
    )
}

#[pyfunction]
#[pyo3(signature=(position_gcrf_m,jd_ut1,jd_tt,xp=0.0,yp=0.0))]
pub(super) fn gcrf_position_to_geodetic(
    py: Python<'_>,
    position_gcrf_m: &PyAny,
    jd_ut1: f64,
    jd_tt: f64,
    xp: f64,
    yp: f64,
) -> PyResult<PyObject> {
    let position = vec3(py, position_gcrf_m, "position_gcrf_m")?;
    pyarr(
        py,
        conversions::gcrf_position_to_geodetic(position, jd_ut1, jd_tt, xp, yp)
            .map_err(PyValueError::new_err)?,
    )
}
