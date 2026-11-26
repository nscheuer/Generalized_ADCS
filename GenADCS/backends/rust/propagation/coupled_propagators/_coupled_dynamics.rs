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
pub fn normalize_q(y: &mut [f64]) -> Result<(), String> {
    if y.len() < 4 {
        return Err("state must contain a quaternion".to_owned());
    }
    let n = y[..4].iter().map(|x| x * x).sum::<f64>().sqrt();
    if n == 0. || !n.is_finite() {
        return Err("quaternion must be finite and nonzero".to_owned());
    }
    for v in &mut y[..4] {
        *v /= n;
    }
    Ok(())
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
    pub fn validate(&self) -> Result<(), String> {
        if !self.mu.is_finite() || self.mu <= 0.0 || !self.radius.is_finite() || self.radius <= 0.0
        {
            return Err("mu and radius must be finite and positive".to_owned());
        }
        if !self.mass.is_finite() || self.mass <= 0.0 {
            return Err("mass must be finite and positive".to_owned());
        }
        if self.normals.len() != self.areas.len()
            || self.arms.len() != self.areas.len()
            || self.drag_coefficients.len() != self.areas.len()
            || self.optical_coefficients.len() != self.areas.len()
        {
            return Err("panel arrays must have matching lengths".to_owned());
        }
        if self.areas.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.drag_coefficients.iter().any(|v| !v.is_finite())
            || self.zonals.iter().any(|v| !v.is_finite())
            || self.normals.iter().flatten().any(|v| !v.is_finite())
            || self.arms.iter().flatten().any(|v| !v.is_finite())
            || self
                .optical_coefficients
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
            || self.dipole.iter().any(|v| !v.is_finite())
            || self.inertia.iter().flatten().any(|v| !v.is_finite())
            || self
                .inverse_inertia
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        {
            return Err(
                "runtime arrays must contain finite values and nonnegative areas".to_owned(),
            );
        }
        if (self.srp_force || self.srp_torque || self.third_body) && self.ephemeris.is_none() {
            return Err("ephemeris is required for SRP and third-body models".to_owned());
        }
        Ok(())
    }
}

pub fn derivative_vec(y: &[f64], jd: f64, p: &Runtime) -> Result<Vec<f64>, String> {
    derivative_array(y, jd, p).map(|values| values.to_vec())
}

fn derivative_array(y: &[f64], jd: f64, p: &Runtime) -> Result<[f64; 13], String> {
    if y.len() != 13 || y.iter().any(|value| !value.is_finite()) {
        return Err("coupled state must contain 13 values".to_owned());
    }
    let mut q = [y[0], y[1], y[2], y[3]];
    normalize_q(&mut q)?;
    let omega = vec3(&y[4..7]);
    let pos = vec3(&y[7..10]);
    let vel = vec3(&y[10..13]);
    let rot = rotation(&q);
    let rot_t = transpose3(rot);
    let mut force = [0.; 3];
    let mut torque = [0.; 3];
    let mut accel = gravity_native::native_acceleration(pos, p.mu, p.radius, &p.zonals)
        .map_err(|error| error.to_string())?;
    if p.drag_force || p.drag_torque {
        let density =
            atmosphere_native::native_density(pos, jd).map_err(|error| error.to_string())?;
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
        let tdb = magnetic_native::native_tdb_jd(jd).map_err(|error| error.to_string())?;
        let eph = p.ephemeris.as_ref().unwrap();
        let sun = scale3(
            eph.position_km(0, tdb).map_err(|error| error.to_string())?,
            1000.,
        );
        if p.third_body {
            let moon = scale3(
                eph.position_km(2, tdb).map_err(|error| error.to_string())?,
                1000.,
            );
            for (body, mu) in [(sun, 1.32712440018e20), (moon, 4.9048695e12)] {
                let rel = sub3(body, pos);
                let d = norm3(body);
                let r = norm3(rel);
                if d == 0. || r == 0. {
                    return Err("third-body distance must be nonzero".to_owned());
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
                return Err("spacecraft and Sun positions coincide".to_owned());
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
        let field = magnetic
            .field_inertial_tesla(pos, jd)
            .map_err(|error| error.to_string())?;
        torque = add3(torque, cross3(p.dipole, mv3(rot, field)));
    }
    let iw = mv3(p.inertia, omega);
    let omegadot = mv3(p.inverse_inertia, sub3(torque, cross3(omega, iw)));
    accel = add3(accel, scale3(force, 1. / p.mass));
    let mut dy = [0.; 13];
    dy[0] = 0.5 * (q[1] * omega[0] + q[2] * omega[1] + q[3] * omega[2]);
    dy[1] = -0.5 * (q[0] * omega[0] + omega[1] * q[3] - omega[2] * q[2]);
    dy[2] = -0.5 * (q[0] * omega[1] + omega[2] * q[1] - omega[0] * q[3]);
    dy[3] = -0.5 * (q[0] * omega[2] + omega[0] * q[2] - omega[1] * q[1]);
    dy[4..7].copy_from_slice(&omegadot);
    dy[7..10].copy_from_slice(&vel);
    dy[10..13].copy_from_slice(&accel);
    Ok(dy)
}
