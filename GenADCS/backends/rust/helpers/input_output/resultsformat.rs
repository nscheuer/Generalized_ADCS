#[pyclass]
#[derive(Clone)]
struct ResultsFormat {
    #[pyo3(get)] attitude_quaternion: bool,
    #[pyo3(get)] satellite_rate: bool,
    #[pyo3(get)] wheel_momenta: bool,
    #[pyo3(get)] N: usize,
}

#[pymethods]
impl ResultsFormat {
    #[new]
    #[pyo3(signature=(attitude_quaternion=true, satellite_rate=true, wheel_momenta=true, N=1))]
    fn new(attitude_quaternion: bool, satellite_rate: bool, wheel_momenta: bool, N: usize) -> PyResult<Self> {
        if N == 0 { return Err(PyValueError::new_err("N must be a positive integer")); }
        Ok(Self { attitude_quaternion, satellite_rate, wheel_momenta, N })
    }
}
