use super::super::*;

#[pyfunction]
pub fn projected_area_drag_force(
    py: Python<'_>,
    relative_velocity_i: &PyAny,
    q_b_from_i: &PyAny,
    density_kg_m3: f64,
    normals_b: &PyAny,
    areas_m2: &PyAny,
    drag_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let velocity = vec3(py, relative_velocity_i, "velocity")?;
    let q = quat(py, q_b_from_i)?;
    let normals = rows(py, normals_b, "normals_b", 3)?
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>();
    let areas = vals(py, areas_m2, "areas")?;
    let cd = vals(py, drag_coefficients, "drag_coefficients")?;
    let (force, _) = super::projected_area::projected_area_drag(
        velocity,
        q,
        density_kg_m3,
        &normals,
        None,
        &areas,
        &cd,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out(py, force))
}

#[pyfunction]
pub fn projected_area_drag_torque(
    py: Python<'_>,
    relative_velocity_i: &PyAny,
    q_b_from_i: &PyAny,
    density_kg_m3: f64,
    normals_b: &PyAny,
    arms_b: &PyAny,
    areas_m2: &PyAny,
    drag_coefficients: &PyAny,
) -> PyResult<PyObject> {
    let velocity = vec3(py, relative_velocity_i, "velocity")?;
    let q = quat(py, q_b_from_i)?;
    let normals = rows(py, normals_b, "normals_b", 3)?
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>();
    let arms = rows(py, arms_b, "arms_b", 3)?
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect::<Vec<_>>();
    let areas = vals(py, areas_m2, "areas")?;
    let cd = vals(py, drag_coefficients, "drag_coefficients")?;
    let (_, torque) = super::projected_area::projected_area_drag(
        velocity,
        q,
        density_kg_m3,
        &normals,
        Some(&arms),
        &areas,
        &cd,
    )
    .map_err(PyValueError::new_err)?;
    Ok(out(py, torque))
}
