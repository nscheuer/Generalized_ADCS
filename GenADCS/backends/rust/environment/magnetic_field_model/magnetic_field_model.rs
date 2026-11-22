#[pyclass(subclass)]
struct MagneticFieldModel;

#[pymethods]
impl MagneticFieldModel {
    #[new]
    fn new() -> Self { Self }
    fn field_inertial_tesla(&self, _position_inertial_m: &PyAny, _epoch_jd: f64) -> PyResult<()> { Err(pyo3::exceptions::PyNotImplementedError::new_err("Choose a magnetic field model")) }
    fn field(&self, _position_inertial_m: &PyAny, _epoch_jd: f64) -> PyResult<()> { Err(pyo3::exceptions::PyNotImplementedError::new_err("Choose a magnetic field model")) }
}

#[pyfunction]
#[pyo3(signature=(position_inertial_m, epoch_jd, g, h, nmax))]
fn magnetic_field_inertial_tesla(py: Python<'_>, position_inertial_m: &PyAny, epoch_jd: f64, g: &PyAny, h: &PyAny, nmax: usize) -> PyResult<PyObject> {
    let position=vec3_from_py(py,position_inertial_m)?;
    let ga:Vec<Vec<Vec<f64>>>=py.import("numpy")?.call_method1("asarray",(g,))?.call_method1("astype",("float64",))?.call_method0("tolist")?.extract().or_else(|_| -> PyResult<Vec<Vec<Vec<f64>>>> {
        let x:Vec<Vec<f64>>=py.import("numpy")?.call_method1("asarray",(g,))?.call_method1("astype",("float64",))?.call_method0("tolist")?.extract()?; Ok(vec![x])
    })?;
    let ha:Vec<Vec<Vec<f64>>>=py.import("numpy")?.call_method1("asarray",(h,))?.call_method1("astype",("float64",))?.call_method0("tolist")?.extract().or_else(|_| -> PyResult<Vec<Vec<Vec<f64>>>> {
        let x:Vec<Vec<f64>>=py.import("numpy")?.call_method1("asarray",(h,))?.call_method1("astype",("float64",))?.call_method0("tolist")?.extract()?; Ok(vec![x])
    })?;
    if ga.is_empty() || ha.is_empty() || ga[0].len()<=nmax || ha[0].len()<=nmax { return Err(PyValueError::new_err("coefficient arrays are too small")); }
    let side=nmax+1;let mut gf=vec![0.;side*side];let mut hf=gf.clone();
    for n in 0..=nmax { for m in 0..=n { gf[n*side+m]=ga[0][n][m];hf[n*side+m]=ha[0][n][m]; } }
    Ok(py_vec3(py,evaluate(position,epoch_jd,&gf,&hf,nmax)?))
}

#[pyclass(extends=MagneticFieldModel)]
struct WMM;
#[pymethods]
impl WMM {
    #[new] fn new() -> (Self, MagneticFieldModel) { (Self,MagneticFieldModel) }
    fn field_inertial_tesla(&self,_position:&PyAny,_epoch_jd:f64)->PyResult<()> { Err(pyo3::exceptions::PyNotImplementedError::new_err("WMM coefficients are not implemented")) }
    fn field(&self,_position:&PyAny,_epoch_jd:f64)->PyResult<()> { Err(pyo3::exceptions::PyNotImplementedError::new_err("WMM coefficients are not implemented")) }
}
