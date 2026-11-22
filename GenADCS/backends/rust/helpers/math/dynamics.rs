#[pyfunction]
fn wheel_momentum_derivative(
    py: Python<'_>,
    tau: &PyAny,
    alpha: &PyAny,
    axes: &PyAny,
) -> PyResult<PyObject> {
    let t = vals(py, tau, "wheel_torque")?;
    let a = fixed::<3>(py, alpha, "angular_acceleration")?;
    let x = py
        .import("numpy")?
        .call_method1("asarray", (axes,))?
        .call_method1("astype", ("float64",))?;
    let rows: Vec<Vec<f64>> = x.call_method0("tolist")?.extract()?;
    if rows.len() != t.len() || rows.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(
            "wheel_axes row count must match wheel_torque size",
        ));
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let axes = DMatrix::from_row_slice(rows.len(), 3, &flat);
    let alpha = SVector::<f64, 3>::from_row_slice(&a);
    let torque = DVector::from_vec(t);
    pyvec(py, (torque - axes * alpha).as_slice().to_vec())
}
#[pyfunction]
fn rigid_body_angular_acceleration(
    py: Python<'_>,
    w: &PyAny,
    t: &PyAny,
    i: &PyAny,
) -> PyResult<PyObject> {
    let w = fixed::<3>(py, w, "angular_velocity")?;
    let t = fixed::<3>(py, t, "torque")?;
    let i = mat3(py, i, "inertia")?;
    let omega = SVector::<f64, 3>::from_row_slice(&w);
    let torque = SVector::<f64, 3>::from_row_slice(&t);
    let rhs = omega.cross(&(i * omega));
    let result = solve3(i, (torque - rhs).into())?;
    pyvec(py, result)
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
    let h = vals(py, h, "wheel_momentum")?;
    let t = fixed::<3>(py, t, "torque")?;
    let i = mat3(py, i, "bus_inertia")?;
    let ar: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (axes,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if ar.len() != h.len() || ar.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err(
            "wheel_axes row count must match wheel_momentum size",
        ));
    }
    let flat: Vec<f64> = ar.iter().flatten().copied().collect();
    let axes = DMatrix::from_row_slice(ar.len(), 3, &flat);
    let wheel_momentum = DVector::from_vec(h);
    let omega = SVector::<f64, 3>::from_row_slice(&w);
    let torque = SVector::<f64, 3>::from_row_slice(&t);
    let total = i * omega + axes.transpose() * wheel_momentum;
    let result = solve3(i, (torque - omega.cross(&total)).into())?;
    pyvec(py, result)
}

fn solve3(a: M, b: [f64; 3]) -> PyResult<[f64; 3]> {
    let rhs = SVector::<f64, 3>::from_row_slice(&b);
    let solution = a
        .full_piv_lu()
        .solve(&rhs)
        .ok_or_else(|| PyValueError::new_err("inertia matrix must be invertible"))?;
    Ok([solution[0], solution[1], solution[2]])
}

const SECS: f64 = 86400.0;
const LEAPS: [(f64, f64); 28] = [
    (2441317.5, 10.),
    (2441499.5, 11.),
    (2441683.5, 12.),
    (2442048.5, 13.),
    (2442413.5, 14.),
    (2442778.5, 15.),
    (2443144.5, 16.),
    (2443509.5, 17.),
    (2443874.5, 18.),
    (2444239.5, 19.),
    (2444786.5, 20.),
    (2445151.5, 21.),
    (2445516.5, 22.),
    (2446247.5, 23.),
    (2447161.5, 24.),
    (2447892.5, 25.),
    (2448257.5, 26.),
    (2448804.5, 27.),
    (2449169.5, 28.),
    (2449534.5, 29.),
    (2450083.5, 30.),
    (2450630.5, 31.),
    (2451179.5, 32.),
    (2453736.5, 33.),
    (2454832.5, 34.),
    (2456109.5, 35.),
    (2457204.5, 36.),
    (2457754.5, 37.),
];
