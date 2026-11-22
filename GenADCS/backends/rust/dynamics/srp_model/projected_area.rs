fn srp_eval(
    py: Python<'_>,
    position: &PyAny,
    sun: &PyAny,
    q: &PyAny,
    normals: &PyAny,
    arms: Option<&PyAny>,
    areas: &PyAny,
    optical: &PyAny,
    illumination: f64,
    torque: bool,
) -> PyResult<PyObject> {
    let p = vec3(py, position, "position_i")?;
    let s = vec3(py, sun, "sun_position_i")?;
    let delta = [s[0] - p[0], s[1] - p[1], s[2] - p[2]];
    let distance = norm(delta);
    if distance == 0. {
        return Err(PyValueError::new_err(
            "spacecraft and Sun positions coincide",
        ));
    }
    let quat = quat(py, q)?;
    let rot = rotation(quat);
    let direction = mv(rot, scale(delta, 1. / distance));
    let pressure = (1361.0 / 299_792_458.0) * (149_597_870_700.0 / distance).powi(2) * illumination;
    let normals = rows(py, normals, "normals_b", 3)?;
    let areas = vals(py, areas, "areas")?;
    let optics = rows(py, optical, "optical_coefficients", 3)?;
    let arms = arms
        .map(|x| rows(py, x, "arms_b", 3))
        .transpose()?
        .unwrap_or_default();
    if normals.len() != areas.len()
        || optics.len() != areas.len()
        || (torque && arms.len() != areas.len())
    {
        return Err(PyValueError::new_err(
            "panel arrays must have matching row counts",
        ));
    }
    let mut total = [0.; 3];
    for i in 0..areas.len() {
        let n = [normals[i][0], normals[i][1], normals[i][2]];
        let cosine = n[0] * direction[0] + n[1] * direction[1] + n[2] * direction[2];
        if cosine > 0. {
            let (spec, diff, abs) = (optics[i][0], optics[i][1], optics[i][2]);
            let term = add(
                scale(direction, abs + diff),
                scale(n, 2. * spec * cosine + 2. * diff / 3.),
            );
            let force = scale(term, -pressure * areas[i] * cosine);
            total = add(
                total,
                if torque {
                    cross([arms[i][0], arms[i][1], arms[i][2]], force)
                } else {
                    force
                },
            );
        }
    }
    if torque {
        Ok(out(py, total))
    } else {
        Ok(out(
            py,
            [
                rot[0][0] * total[0] + rot[1][0] * total[1] + rot[2][0] * total[2],
                rot[0][1] * total[0] + rot[1][1] * total[1] + rot[2][1] * total[2],
                rot[0][2] * total[0] + rot[1][2] * total[1] + rot[2][2] * total[2],
            ],
        ))
    }
}
#[pyfunction]
#[pyo3(signature=(position_i,sun_position_i,q_b_from_i,normals_b,areas,optical_coefficients,illumination=1.0))]
fn projected_area_srp_force(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    q_b_from_i: &PyAny,
    normals_b: &PyAny,
    areas: &PyAny,
    optical_coefficients: &PyAny,
    illumination: f64,
) -> PyResult<PyObject> {
    srp_eval(
        py,
        position_i,
        sun_position_i,
        q_b_from_i,
        normals_b,
        None,
        areas,
        optical_coefficients,
        illumination,
        false,
    )
}
#[pyfunction]
#[pyo3(signature=(position_i,sun_position_i,q_b_from_i,normals_b,arms_b,areas,optical_coefficients,illumination=1.0))]
fn projected_area_srp_torque(
    py: Python<'_>,
    position_i: &PyAny,
    sun_position_i: &PyAny,
    q_b_from_i: &PyAny,
    normals_b: &PyAny,
    arms_b: &PyAny,
    areas: &PyAny,
    optical_coefficients: &PyAny,
    illumination: f64,
) -> PyResult<PyObject> {
    srp_eval(
        py,
        position_i,
        sun_position_i,
        q_b_from_i,
        normals_b,
        Some(arms_b),
        areas,
        optical_coefficients,
        illumination,
        true,
    )
}
