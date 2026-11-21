fn numbers(py: Python<'_>, x: &PyAny) -> PyResult<Vec<f64>> {
    py.import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()
}
fn matrix3(py: Python<'_>, x: &PyAny) -> PyResult<[[f64; 3]; 3]> {
    let a: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    if a.len() != 3 || a.iter().any(|r| r.len() != 3) {
        return Err(PyValueError::new_err("matrix must be 3x3"));
    }
    Ok([
        a[0].clone().try_into().unwrap(),
        a[1].clone().try_into().unwrap(),
        a[2].clone().try_into().unwrap(),
    ])
}
fn rows3(py: Python<'_>, x: &PyAny) -> PyResult<Vec<[f64; 3]>> {
    let a: Vec<Vec<f64>> = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?
        .call_method0("tolist")?
        .extract()?;
    a.into_iter()
        .map(|r| {
            r.try_into()
                .map_err(|_| PyValueError::new_err("panel arrays must have three columns"))
        })
        .collect()
}
fn vec3(a: &[f64]) -> [f64; 3] {
    [a[0], a[1], a[2]]
}
fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn norm3(a: [f64; 3]) -> f64 {
    dot3(a, a).sqrt()
}
fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale3(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn mv3(a: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [dot3(a[0], v), dot3(a[1], v), dot3(a[2], v)]
}
fn transpose3(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [a[0][0], a[1][0], a[2][0]],
        [a[0][1], a[1][1], a[2][1]],
        [a[0][2], a[1][2], a[2][2]],
    ]
}
fn rotation(q: &[f64]) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    [
        [
            w * w + x * x - y * y - z * z,
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            w * w - x * x + y * y - z * z,
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            w * w - x * x - y * y + z * z,
        ],
    ]
}
fn normalize_q(y: &mut [f64]) -> PyResult<()> {
    if y.len() < 4 {
        return Err(PyValueError::new_err("state must contain a quaternion"));
    }
    let n = y[..4].iter().map(|x| x * x).sum::<f64>().sqrt();
    if n == 0. || !n.is_finite() {
        return Err(PyValueError::new_err(
            "quaternion must be finite and nonzero",
        ));
    }
    for v in &mut y[..4] {
        *v /= n;
    }
    Ok(())
}
fn py_array(py: Python<'_>, v: &[f64]) -> PyObject {
    v.to_pyarray(py).into_py(py)
}

pub struct Runtime {
    pub mu: f64,
    pub radius: f64,
    pub mass: f64,
    pub inertia: [[f64; 3]; 3],
    pub inverse_inertia: [[f64; 3]; 3],
    pub zonals: Vec<f64>,
    pub normals: Vec<[f64; 3]>,
    pub arms: Vec<[f64; 3]>,
    pub areas: Vec<f64>,
    pub drag_coefficients: Vec<f64>,
    pub optical_coefficients: Vec<[f64; 3]>,
    pub dipole: [f64; 3],
    pub drag_force: bool,
    pub drag_torque: bool,
    pub srp_force: bool,
    pub srp_torque: bool,
    pub gravity_gradient: bool,
    pub third_body: bool,
    pub ephemeris: Option<ephemeris_native::NativeEphemeris>,
    pub magnetic: Option<magnetic_native::NativeMagneticField>,
}
impl Runtime {
    fn from_py(py: Python<'_>, p: &PyAny) -> PyResult<Self> {
        let normals = rows3(py, p.getattr("normals")?)?;
        let arms = rows3(py, p.getattr("arms")?)?;
        let areas = numbers(py, p.getattr("areas")?)?;
        let drag_coefficients = numbers(py, p.getattr("drag_coefficients")?)?;
        let optical_coefficients = rows3(py, p.getattr("optical_coefficients")?)?;
        let dipole = numbers(py, p.getattr("dipole")?)?;
        if dipole.len() != 3
            || [
                normals.len(),
                arms.len(),
                drag_coefficients.len(),
                optical_coefficients.len(),
            ]
            .iter()
            .any(|n| *n != areas.len())
        {
            return Err(PyValueError::new_err("invalid panel or dipole dimensions"));
        }
        let srp_force = p.getattr("srp_force")?.extract()?;
        let srp_torque = p.getattr("srp_torque")?.extract()?;
        let third_body = p.getattr("third_body")?.extract()?;
        let residual_dipole: bool = p.getattr("residual_dipole")?.extract()?;
        let ephemeris = if srp_force || srp_torque || third_body {
            let path: String = match p.getattr("ephemeris_path") {
                Ok(value) => value.extract()?,
                Err(_) => concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../environment/ephemeris/de421/de421.bsp"
                )
                .to_owned(),
            };
            Some(ephemeris_native::NativeEphemeris::from_path(
                std::path::Path::new(&path),
            )?)
        } else {
            None
        };
        let magnetic = if residual_dipole {
            let kind: i32 = p.getattr("magnetic_kind")?.extract()?;
            Some(match kind {
                1 => magnetic_native::NativeMagneticField::igrf14()?,
                2 => magnetic_native::NativeMagneticField::chaos8(
                    p.getattr("magnetic_include_static")?.extract()?,
                )?,
                _ => return Err(PyValueError::new_err("unsupported magnetic model")),
            })
        } else {
            None
        };
        Ok(Self {
            mu: p.getattr("mu")?.extract()?,
            radius: p.getattr("radius")?.extract()?,
            mass: p.getattr("mass")?.extract()?,
            inertia: matrix3(py, p.getattr("inertia")?)?,
            inverse_inertia: matrix3(py, p.getattr("inverse_inertia")?)?,
            zonals: numbers(py, p.getattr("zonals")?)?,
            normals,
            arms,
            areas,
            drag_coefficients,
            optical_coefficients,
            dipole: vec3(&dipole),
            drag_force: p.getattr("drag_force")?.extract()?,
            drag_torque: p.getattr("drag_torque")?.extract()?,
            srp_force,
            srp_torque,
            gravity_gradient: p.getattr("gravity_gradient")?.extract()?,
            third_body,
            ephemeris,
            magnetic,
        })
    }
}
pub fn derivative_vec(y: &[f64], jd: f64, p: &Runtime) -> PyResult<Vec<f64>> {
    if y.len() != 13 {
        return Err(PyValueError::new_err(
            "coupled state must contain 13 values",
        ));
    }
    let mut q = y[..4].to_vec();
    normalize_q(&mut q)?;
    let omega = vec3(&y[4..7]);
    let pos = vec3(&y[7..10]);
    let vel = vec3(&y[10..13]);
    let rot = rotation(&q);
    let rot_t = transpose3(rot);
    let mut force = [0.; 3];
    let mut torque = [0.; 3];
    let mut accel = gravity_native::native_acceleration(pos, p.mu, p.radius, &p.zonals)?;
    if p.drag_force || p.drag_torque {
        let density = atmosphere_native::native_density(pos, jd)?;
        let relative = sub3(vel, atmosphere_native::native_corotation(pos));
        let v = mv3(rot, relative);
        let speed = norm3(v);
        if speed > 0. && density > 0. {
            for i in 0..p.areas.len() {
                let incidence = dot3(p.normals[i], v) / speed;
                if incidence > 0. {
                    let panel = scale3(
                        v,
                        -0.5 * density * speed * p.areas[i] * p.drag_coefficients[i] * incidence,
                    );
                    if p.drag_force {
                        force = add3(force, mv3(rot_t, panel));
                    }
                    if p.drag_torque {
                        torque = add3(torque, cross3(p.arms[i], panel));
                    }
                }
            }
        }
    }
    if p.srp_force || p.srp_torque || p.third_body {
        let tdb = magnetic_native::native_tdb_jd(jd)?;
        let eph = p.ephemeris.as_ref().unwrap();
        let sun = scale3(eph.position_km(0, tdb)?, 1000.);
        if p.third_body {
            let moon = scale3(eph.position_km(2, tdb)?, 1000.);
            for (body, mu) in [(sun, 1.32712440018e20), (moon, 4.9048695e12)] {
                let rel = sub3(body, pos);
                let d = norm3(body);
                let r = norm3(rel);
                if d == 0. || r == 0. {
                    return Err(PyValueError::new_err("third-body distance must be nonzero"));
                }
                accel = add3(
                    accel,
                    sub3(scale3(rel, mu / r.powi(3)), scale3(body, mu / d.powi(3))),
                );
            }
        }
        if p.srp_force || p.srp_torque {
            let delta = sub3(sun, pos);
            let distance = norm3(delta);
            if distance == 0. {
                return Err(PyValueError::new_err(
                    "spacecraft and Sun positions coincide",
                ));
            }
            let direction = mv3(rot, scale3(delta, 1. / distance));
            let illumination = ephemeris_native::native_eclipse(pos, sun, p.radius, 696340000.).0;
            let pressure = (1361. / 299792458.) * (149597870700. / distance).powi(2) * illumination;
            for i in 0..p.areas.len() {
                let normal = p.normals[i];
                let cosine = dot3(normal, direction);
                if cosine > 0. {
                    let [spec, diff, abs] = p.optical_coefficients[i];
                    let panel = scale3(
                        add3(
                            scale3(direction, abs + diff),
                            scale3(normal, 2. * spec * cosine + 2. * diff / 3.),
                        ),
                        -pressure * p.areas[i] * cosine,
                    );
                    if p.srp_force {
                        force = add3(force, mv3(rot_t, panel));
                    }
                    if p.srp_torque {
                        torque = add3(torque, cross3(p.arms[i], panel));
                    }
                }
            }
        }
    }
    if p.gravity_gradient {
        let r = norm3(pos);
        let radial = mv3(rot, scale3(pos, 1. / r));
        torque = add3(
            torque,
            scale3(
                cross3(radial, mv3(p.inertia, radial)),
                3. * p.mu / r.powi(3),
            ),
        );
    }
    if let Some(magnetic) = &p.magnetic {
        let field = magnetic.field_inertial_tesla(pos, jd)?;
        torque = add3(torque, cross3(p.dipole, mv3(rot, field)));
    }
    let iw = mv3(p.inertia, omega);
    let omegadot = mv3(p.inverse_inertia, sub3(torque, cross3(omega, iw)));
    accel = add3(accel, scale3(force, 1. / p.mass));
    let mut dy = vec![0.; 13];
    dy[0] = 0.5 * (q[1] * omega[0] + q[2] * omega[1] + q[3] * omega[2]);
    dy[1] = -0.5 * (q[0] * omega[0] + omega[1] * q[3] - omega[2] * q[2]);
    dy[2] = -0.5 * (q[0] * omega[1] + omega[2] * q[1] - omega[0] * q[3]);
    dy[3] = -0.5 * (q[0] * omega[2] + omega[0] * q[2] - omega[1] * q[1]);
    dy[4..7].copy_from_slice(&omegadot);
    dy[7..10].copy_from_slice(&vel);
    dy[10..13].copy_from_slice(&accel);
    Ok(dy)
}
#[pyfunction]
fn derivative(py: Python<'_>, y: &PyAny, jd: f64, runtime: &PyAny) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    let r = Runtime::from_py(py, runtime)?;
    Ok(py_array(py, &derivative_vec(&y, jd, &r)?))
}
#[pyfunction]
fn project(py: Python<'_>, y: &PyAny, _runtime: &PyAny) -> PyResult<PyObject> {
    let y = numbers(py, y)?;
    if y.len() != 13 {
        return Err(PyValueError::new_err(
            "coupled state must contain 13 values",
        ));
    }
    Ok(py_array(py, &y))
}
