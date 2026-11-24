use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyIOError, PyValueError};
use std::path::PathBuf;
use std::sync::Arc;

static DEFAULT_KERNEL: OnceCell<Arc<Kernel>> = OnceCell::new();

fn default_kernel_path(py: Python<'_>) -> PyResult<PathBuf> {
    let package_file: String = py.import("GenADCS")?.getattr("__file__")?.extract()?;
    let package_dir = Path::new(&package_file).parent().ok_or_else(|| {
        PyIOError::new_err("cannot locate GenADCS package directory")
    })?;
    Ok(package_dir.join("kernels/environment/ephemeris/de421/de421.bsp"))
}

fn bundled_kernel(py: Python<'_>) -> PyResult<Arc<Kernel>> {
    DEFAULT_KERNEL
        .get_or_try_init(|| Kernel::load(&default_kernel_path(py)?).map(Arc::new).map_err(PyValueError::new_err))
        .map(Arc::clone)
}

fn kernel_from_optional_arrays(
    py: Python<'_>, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>,
) -> PyResult<Arc<Kernel>> {
    match (segments, times, data) {
        (None, None, None) => bundled_kernel(py),
        (Some(s), Some(t), Some(d)) => Ok(Arc::new(kernel_from_arrays(py, s, t, d)?)),
        _ => Err(PyValueError::new_err("segments, times, and data must be supplied together")),
    }
}

fn kernel_from_arrays(py: Python<'_>, segments: &PyAny, times: &PyAny, data: &PyAny) -> PyResult<Kernel> {
    let segment_rows: Vec<Vec<i64>> = py.import("numpy")?.call_method1("asarray", (segments,))?.call_method1("astype", ("int64",))?.call_method0("tolist")?.extract()?;
    let time_rows: Vec<Vec<f64>> = py.import("numpy")?.call_method1("asarray", (times,))?.call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    let data: Vec<f64> = py.import("numpy")?.call_method1("asarray", (data,))?.call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    if segment_rows.len() != time_rows.len() || segment_rows.iter().any(|r| r.len() != 5) || time_rows.iter().any(|r| r.len() != 2) || data.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err("invalid parsed SPK segment tables"));
    }
    let segments = segment_rows.into_iter().zip(time_rows).map(|(s,t)| Segment { target:s[0] as i32, center:s[1] as i32, kind:s[2] as i32, offset:s[3] as usize, size:s[4] as usize, start:t[0], end:t[1] }).collect();
    Ok(Kernel { segments, data })
}

fn body_index(body: &PyAny) -> PyResult<usize> {
    if let Ok(name) = body.extract::<String>() {
        return body_name_index(&name);
    }
    let id = body.extract::<i32>()?;
    match id {
        0 => Ok(10), 1 => Ok(11), 2 => Ok(9),
        199 => Ok(0), 299 => Ok(1), 3 => Ok(2),
        4 | 499 => Ok(3), 5 | 599 => Ok(4), 6 | 699 => Ok(5),
        7 | 799 => Ok(6), 8 | 899 => Ok(7), 9 | 999 => Ok(8),
        301 => Ok(9), 10 => Ok(10), 399 => Ok(11),
        _ => Err(PyValueError::new_err(format!("unsupported DE421 body identifier {id}"))),
    }
}

fn body_name_index(body: &str) -> PyResult<usize> {
    let name = body.to_lowercase();
    match name.as_str() {
            "mercury" => Ok(0), "venus" => Ok(1), "earth-moon barycenter" | "emb" => Ok(2),
            "mars" => Ok(3), "jupiter" => Ok(4), "saturn" => Ok(5), "uranus" => Ok(6),
            "neptune" => Ok(7), "pluto" => Ok(8), "moon" => Ok(9), "sun" => Ok(10), "earth" => Ok(11),
            _ => Err(PyValueError::new_err(format!("unsupported DE421 body {name:?}"))),
    }
}

fn vec3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py.import("numpy")?.call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    if values.len() != 3 || values.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must be a finite 3-vector")));
    }
    Ok([values[0], values[1], values[2]])
}

#[pyfunction(name = "eclipse_state")]
fn eclipse_state_py(py: Python<'_>, spacecraft_position_m: &PyAny, sun_position_m: &PyAny, occulting_body_radius_m: f64, sun_radius_m: f64) -> PyResult<(f64, bool, bool)> {
    Ok(eclipse_calc(vec3(py, spacecraft_position_m, "spacecraft_position_m")?, vec3(py, sun_position_m, "sun_position_m")?, occulting_body_radius_m, sun_radius_m))
}

#[pyfunction]
#[pyo3(signature = (body_id, epoch_jd, segments=None, times=None, data=None))]
fn ephemeris_position(py: Python<'_>, body_id: i32, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let index = match body_id { 0=>10, 1=>11, 2=>9, _=>return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")) };
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(index, epoch_jd).map_err(PyValueError::new_err)?.position.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_id, epoch_jd, segments=None, times=None, data=None))]
fn ephemeris_velocity(py: Python<'_>, body_id: i32, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let index = match body_id { 0=>10, 1=>11, 2=>9, _=>return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")) };
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(index, epoch_jd).map_err(PyValueError::new_err)?.velocity.to_pyarray(py).into_py(py))
}

#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_position(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(body_index, epoch_jd).map_err(PyValueError::new_err)?.position.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_velocity(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(body_index, epoch_jd).map_err(PyValueError::new_err)?.velocity.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_state(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<(PyObject, PyObject)> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    let state = kernel.planetary_state(body_index, epoch_jd).map_err(PyValueError::new_err)?;
    Ok((state.position.to_pyarray(py).into_py(py), state.velocity.to_pyarray(py).into_py(py)))
}

