fn parse(name: &str) -> PyResult<u8> {
    match name.to_uppercase().as_str() {
        "GCRF" => Ok(0),
        "ECEF" => Ok(1),
        "BODY" => Ok(2),
        "LVLH" => Ok(3),
        "TEME" => Ok(4),
        _ => Err(PyValueError::new_err(format!(
            "unsupported frame {name}; expected GCRF, ECEF, BODY, LVLH, or TEME"
        ))),
    }
}
fn from_gcrf(
    kind: u8,
    jd_ut1: Option<f64>,
    jd_tt: Option<f64>,
    xp: f64,
    yp: f64,
    p: Option<V>,
    v: Option<V>,
    q: Option<[f64; 4]>,
) -> PyResult<M> {
    match kind {
        0 => Ok(M::identity()),
        1 => {
            let u = jd_ut1.ok_or_else(|| {
                PyValueError::new_err("jd_ut1 is required for ECEF or TEME conversions")
            })?;
            let t = jd_tt.ok_or_else(|| {
                PyValueError::new_err("jd_tt is required for ECEF or TEME conversions")
            })?;
            Ok(mm(teme_ecef(u, xp, yp), gcrf_teme(t)))
        }
        4 => {
            let _u = jd_ut1.ok_or_else(|| {
                PyValueError::new_err("jd_ut1 is required for ECEF or TEME conversions")
            })?;
            let t = jd_tt.ok_or_else(|| {
                PyValueError::new_err("jd_tt is required for ECEF or TEME conversions")
            })?;
            Ok(gcrf_teme(t))
        }
        3 => lvlh(
            p.ok_or_else(|| {
                PyValueError::new_err(
                    "position_gcrf and velocity_gcrf are required for LVLH conversions",
                )
            })?,
            v.ok_or_else(|| {
                PyValueError::new_err(
                    "position_gcrf and velocity_gcrf are required for LVLH conversions",
                )
            })?,
        ).map_err(PyValueError::new_err),
        _ => Ok(qmat(q.ok_or_else(|| {
            PyValueError::new_err("q_body_from_gcrf is required for body-frame conversions")
        })?)),
    }
}
fn context_rotation(
    source: &str,
    target: &str,
    jdu: Option<f64>,
    jdt: Option<f64>,
    xp: f64,
    yp: f64,
    p: Option<V>,
    v: Option<V>,
    q: Option<[f64; 4]>,
) -> PyResult<M> {
    let s = parse(source)?;
    let t = parse(target)?;
    if s == t {
        return Ok(M::identity());
    }
    let a = from_gcrf(t, jdu, jdt, xp, yp, p, v, q)?;
    let b = from_gcrf(s, jdu, jdt, xp, yp, p, v, q)?;
    Ok(mm(a, tr(b)))
}
fn optional_fixed3(py: Python<'_>, x: Option<&PyAny>, name: &str) -> PyResult<Option<V>> {
    x.map(|a| vec3(py, a, name)).transpose()
}
fn optional_q(py: Python<'_>, x: Option<&PyAny>) -> PyResult<Option<[f64; 4]>> {
    x.map(|a| {
        let r: Vec<f64> = py
            .import("numpy")?
            .call_method1("asarray", (a,))?
            .call_method1("astype", ("float64",))?
            .call_method0("tolist")?
            .extract()?;
        let r: [f64; 4] = r
            .try_into()
            .map_err(|_| PyValueError::new_err("q_body_from_gcrf has an invalid shape"))?;
        if r.iter().any(|v| !v.is_finite()) {
            return Err(PyValueError::new_err(
                "q_body_from_gcrf must contain only finite values",
            ));
        }
        Ok(r)
    })
    .transpose()
}
#[pyfunction]
#[pyo3(signature=(source,target,*,jd_ut1=None,jd_tt=None,xp=0.0,yp=0.0,position_gcrf=None,velocity_gcrf=None,q_body_from_gcrf=None))]
fn frame_rotation(
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
    let p = optional_fixed3(py, position_gcrf, "position_gcrf")?;
    let v = optional_fixed3(py, velocity_gcrf, "velocity_gcrf")?;
    let q = optional_q(py, q_body_from_gcrf)?;
    Ok(pymat(
        py,
        &context_rotation(source, target, jd_ut1, jd_tt, xp, yp, p, v, q)?,
    ))
}
#[pyfunction]
#[pyo3(signature=(vector,source,target,*,jd_ut1=None,jd_tt=None,xp=0.0,yp=0.0,position_gcrf=None,velocity_gcrf=None,q_body_from_gcrf=None))]
fn convert_vector(
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
    let p = optional_fixed3(py, position_gcrf, "position_gcrf")?;
    let v = optional_fixed3(py, velocity_gcrf, "velocity_gcrf")?;
    let q = optional_q(py, q_body_from_gcrf)?;
    pyarr(
        py,
        mv(
            context_rotation(source, target, jd_ut1, jd_tt, xp, yp, p, v, q)?,
            vec3(py, vector, "vector")?,
        ),
    )
}
#[pyfunction]
fn ecef_to_geodetic(py: Python<'_>, position_ecef_m: &PyAny) -> PyResult<PyObject> {
    pyarr(py, ecef_geo(vec3(py, position_ecef_m, "position_ecef_m")?))
}
#[pyfunction]
#[pyo3(signature=(position_gcrf_m,jd_ut1,jd_tt,xp=0.0,yp=0.0))]
fn gcrf_position_to_geodetic(
    py: Python<'_>,
    position_gcrf_m: &PyAny,
    jd_ut1: f64,
    jd_tt: f64,
    xp: f64,
    yp: f64,
) -> PyResult<PyObject> {
    let p = vec3(py, position_gcrf_m, "position_gcrf_m")?;
    let ecef = mv(mm(teme_ecef(jd_ut1, xp, yp), gcrf_teme(jd_tt)), p);
    pyarr(py, ecef_geo(ecef))
}
