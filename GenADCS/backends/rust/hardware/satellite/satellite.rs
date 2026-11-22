#[pyclass]
struct Satellite {
    #[pyo3(get,set)] geometry:PyObject,
    #[pyo3(get,set)] adcs:PyObject,
    #[pyo3(get,set)] state_machine:PyObject,
}
#[pymethods]
impl Satellite {
    #[new]
    #[pyo3(signature=(geometry,adcs=None,state_machine=None))]
    fn new(py:Python<'_>,geometry:PyObject,adcs:Option<PyObject>,state_machine:Option<PyObject>)->PyResult<Self>{
        let adcs=match adcs{Some(value) if !value.is_none(py)=>value,_=>py.import("GenADCS.backends.rust.hardware.ADCS_subsystem")?.getattr("ADCSSubsystem")?.call0()?.into_py(py)};
        Ok(Self{geometry,adcs,state_machine:state_machine.unwrap_or_else(||py.None())})
    }
}
