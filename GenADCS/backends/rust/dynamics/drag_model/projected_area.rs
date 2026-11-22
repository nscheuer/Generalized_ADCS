fn drag_eval(
    py: Python<'_>,
    velocity: &PyAny,
    q: &PyAny,
    density: f64,
    normals: &PyAny,
    arms: Option<&PyAny>,
    areas: &PyAny,
    cd: &PyAny,
    torque: bool,
) -> PyResult<PyObject> {
    let velocity = mv(rotation(quat(py, q)?), vec3(py, velocity, "velocity")?);
    let speed = norm(velocity);
    let normals = rows(py, normals, "normals_b", 3)?;
    let areas = vals(py, areas, "areas")?;
    let cd = vals(py, cd, "drag_coefficients")?;
    let arms = arms
        .map(|v| rows(py, v, "arms_b", 3))
        .transpose()?
        .unwrap_or_default();
    if normals.len() != areas.len()
        || cd.len() != areas.len()
        || (torque && arms.len() != areas.len())
    {
        return Err(PyValueError::new_err(
            "panel arrays must have matching row counts",
        ));
    }
    let mut sum = [0.; 3];
    if speed > 0. && density > 0. {
        for i in 0..areas.len() {
            let incidence = normals[i][0] * velocity[0]
                + normals[i][1] * velocity[1]
                + normals[i][2] * velocity[2];
            let incidence = incidence / speed;
            if incidence > 0. {
                let f = scale(
                    velocity,
                    -0.5 * density * speed * areas[i] * cd[i] * incidence,
                );
                sum = add(
                    sum,
                    if torque {
                        cross([arms[i][0], arms[i][1], arms[i][2]], f)
                    } else {
                        f
                    },
                );
            }
        }
    }
    if torque {
        Ok(out(py, sum))
    } else {
        let r = rotation(quat(py, q)?);
        Ok(out(
            py,
            [
                r[0][0] * sum[0] + r[1][0] * sum[1] + r[2][0] * sum[2],
                r[0][1] * sum[0] + r[1][1] * sum[1] + r[2][1] * sum[2],
                r[0][2] * sum[0] + r[1][2] * sum[1] + r[2][2] * sum[2],
            ],
        ))
    }
}
#[pyfunction]
fn projected_area_drag_force(
    py: Python<'_>,
    relative_velocity_i: &PyAny,
    q_b_from_i: &PyAny,
    density_kg_m3: f64,
    normals_b: &PyAny,
    areas_m2: &PyAny,
    drag_coefficients: &PyAny,
) -> PyResult<PyObject> {
    drag_eval(
        py,
        relative_velocity_i,
        q_b_from_i,
        density_kg_m3,
        normals_b,
        None,
        areas_m2,
        drag_coefficients,
        false,
    )
}
#[pyfunction]
fn projected_area_drag_torque(
    py: Python<'_>,
    relative_velocity_i: &PyAny,
    q_b_from_i: &PyAny,
    density_kg_m3: f64,
    normals_b: &PyAny,
    arms_b: &PyAny,
    areas_m2: &PyAny,
    drag_coefficients: &PyAny,
) -> PyResult<PyObject> {
    drag_eval(
        py,
        relative_velocity_i,
        q_b_from_i,
        density_kg_m3,
        normals_b,
        Some(arms_b),
        areas_m2,
        drag_coefficients,
        true,
    )
}
