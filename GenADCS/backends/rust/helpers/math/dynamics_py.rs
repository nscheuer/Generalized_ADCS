fn wheel_axes(py: Python<'_>, value: &PyAny, expected: usize) -> PyResult<Vec<[f64; 3]>> {
    let rows: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (value,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if rows.len() != expected || rows.iter().any(|row| row.len() != 3) {
        return Err(PyValueError::new_err(
            "wheel_axes row count must match wheel vector size",
        ));
    }
    rows.into_iter()
        .map(|row| {
            row.try_into()
                .map_err(|_| PyValueError::new_err("wheel_axes must have three columns"))
        })
        .collect()
}

#[pyfunction]
fn wheel_momentum_derivative(
    py: Python<'_>,
    tau: &PyAny,
    alpha: &PyAny,
    axes: &PyAny,
) -> PyResult<PyObject> {
    let torque = vals(py, tau, "wheel_torque")?;
    let alpha = fixed::<3>(py, alpha, "angular_acceleration")?;
    let axes = wheel_axes(py, axes, torque.len())?;
    pyvec(
        py,
        wheel_momentum_derivative_native(&torque, alpha, &axes).map_err(PyValueError::new_err)?,
    )
}

#[pyfunction]
fn rigid_body_angular_acceleration(
    py: Python<'_>,
    w: &PyAny,
    t: &PyAny,
    i: &PyAny,
) -> PyResult<PyObject> {
    let w = fixed::<3>(py, w, "angular_velocity")?;
    let torque = fixed::<3>(py, t, "torque")?;
    let inertia = mat3(py, i, "inertia")?;
    pyvec(
        py,
        rigid_body_angular_acceleration_native(w, torque, inertia)
            .map_err(PyValueError::new_err)?,
    )
}

#[pyfunction]
fn wheel_coupled_angular_acceleration(
    py: Python<'_>,
    w: &PyAny,
    h: &PyAny,
    axes: &PyAny,
    t: &PyAny,
    i: &PyAny,
) -> PyResult<PyObject> {
    let w = fixed::<3>(py, w, "angular_velocity")?;
    let momentum = vals(py, h, "wheel_momentum")?;
    let axes = wheel_axes(py, axes, momentum.len())?;
    let torque = fixed::<3>(py, t, "torque")?;
    let inertia = mat3(py, i, "bus_inertia")?;
    pyvec(
        py,
        wheel_coupled_angular_acceleration_native(w, &momentum, &axes, torque, inertia)
            .map_err(PyValueError::new_err)?,
    )
}
