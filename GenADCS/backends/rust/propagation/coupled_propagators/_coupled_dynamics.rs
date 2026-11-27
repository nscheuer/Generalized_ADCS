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
    pub wheel_axes: Vec<[f64; 3]>,
    pub wheel_inertias: Vec<f64>,
    pub inverse_bus_inertia: [[f64; 3]; 3],
    pub wheel_commands: Vec<f64>,
    pub wheel_max_momentum: Vec<f64>,
    pub mtq_axes: Vec<[f64; 3]>,
    pub mtq_max_dipole: Vec<f64>,
    pub dipole_commands: Vec<f64>,
}

impl Runtime {
    pub fn state_len(&self) -> usize {
        13 + self.wheel_commands.len()
    }

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
        if self.wheel_axes.len() != self.wheel_inertias.len()
            || self.wheel_axes.len() != self.wheel_commands.len()
            || self.wheel_axes.len() != self.wheel_max_momentum.len()
        {
            return Err("reaction wheel arrays must have matching lengths".to_owned());
        }
        if self.wheel_axes.iter().flatten().any(|v| !v.is_finite())
            || self
                .wheel_inertias
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            || self
                .inverse_bus_inertia
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
            || self.wheel_commands.iter().any(|v| !v.is_finite())
            || self
                .wheel_max_momentum
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(
                "reaction wheel parameters must be finite and inertias/limits positive".to_owned(),
            );
        }
        Ok(())
    }
}

pub fn project_wheel_momentum(y: &mut [f64], p: &Runtime) -> Result<(), String> {
    let expected = p.state_len();
    if y.len() != expected {
        return Err(format!("coupled state must contain {expected} values"));
    }
    for (wheel, &limit) in p.wheel_max_momentum.iter().enumerate() {
        if y[13 + wheel] > limit {
            y[13 + wheel] = limit;
        } else if y[13 + wheel] < -limit {
            y[13 + wheel] = -limit;
        }
    }
    Ok(())
}

pub fn derivative_vec(y: &[f64], jd: f64, p: &Runtime) -> Result<Vec<f64>, String> {
    let expected = p.state_len();
    if y.len() != expected || y.iter().any(|value| !value.is_finite()) {
        return Err(format!("coupled state must contain {expected} values"));
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
        let b_body = mv3(rot, field);
        torque = add3(torque, cross3(p.dipole, b_body));
        if !p.dipole_commands.is_empty() {
            let mut mtq_dipole = [0.0; 3];
            for (axis, &cmd) in p.mtq_axes.iter().zip(&p.dipole_commands) {
                mtq_dipole = add3(mtq_dipole, scale3(*axis, cmd));
            }
            torque = add3(torque, cross3(mtq_dipole, b_body));
        }
    }
    for (axis, &command) in p.wheel_axes.iter().zip(&p.wheel_commands) {
        torque = add3(torque, scale3(*axis, command));
    }
    let mut angular_momentum = mv3(p.inertia, omega);
    let omegadot = if p.wheel_axes.is_empty() {
        mv3(p.inverse_inertia, sub3(torque, cross3(omega, angular_momentum)))
    } else {
        for (wheel, axis) in p.wheel_axes.iter().enumerate() {
            angular_momentum = add3(angular_momentum, scale3(*axis, y[13 + wheel]));
        }
        mv3(
            p.inverse_bus_inertia,
            sub3(torque, cross3(omega, angular_momentum)),
        )
    };
    accel = add3(accel, scale3(force, 1. / p.mass));
    let mut dy = vec![0.; expected];
    dy[0] = 0.5 * (q[1] * omega[0] + q[2] * omega[1] + q[3] * omega[2]);
    dy[1] = -0.5 * (q[0] * omega[0] + omega[1] * q[3] - omega[2] * q[2]);
    dy[2] = -0.5 * (q[0] * omega[1] + omega[2] * q[1] - omega[0] * q[3]);
    dy[3] = -0.5 * (q[0] * omega[2] + omega[0] * q[2] - omega[1] * q[1]);
    dy[4..7].copy_from_slice(&omegadot);
    dy[7..10].copy_from_slice(&vel);
    dy[10..13].copy_from_slice(&accel);
    for wheel in 0..p.wheel_commands.len() {
        let mut momentum_rate =
            -p.wheel_commands[wheel] - p.wheel_inertias[wheel] * dot3(p.wheel_axes[wheel], omegadot);
        let momentum = y[13 + wheel];
        let limit = p.wheel_max_momentum[wheel];
        if (momentum >= limit && momentum_rate > 0.0)
            || (momentum <= -limit && momentum_rate < 0.0)
        {
            momentum_rate = 0.0;
        }
        dy[13 + wheel] = momentum_rate;
    }
    Ok(dy)
}

#[cfg(test)]
mod rw_tests {
    use super::*;

    fn test_runtime_with_wheel() -> Runtime {
        Runtime {
            mu: 3.986004418e14,
            radius: 6_378_137.0,
            mass: 10.0,
            inertia: [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]],
            inverse_inertia: [[0.5, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 0.25]],
            zonals: vec![0.0; 5],
            normals: Vec::new(),
            arms: Vec::new(),
            areas: Vec::new(),
            drag_coefficients: Vec::new(),
            optical_coefficients: Vec::new(),
            dipole: [0.0; 3],
            drag_force: false,
            drag_torque: false,
            srp_force: false,
            srp_torque: false,
            gravity_gradient: false,
            third_body: false,
            ephemeris: None,
            magnetic: None,
            wheel_axes: vec![[0.0, 0.0, 1.0]],
            wheel_inertias: vec![0.1],
            inverse_bus_inertia: [[0.5, 0.0, 0.0], [0.0, 1.0 / 3.0, 0.0], [0.0, 0.0, 1.0 / 3.9]],
            wheel_commands: vec![0.05],
            wheel_max_momentum: vec![0.5],
            mtq_axes: Vec::new(),
            mtq_max_dipole: Vec::new(),
            dipole_commands: Vec::new(),
        }
    }

    #[test]
    fn projects_wheel_momentum_within_limits() {
        let runtime = test_runtime_with_wheel();
        let mut y = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0, 1.2,
        ];
        project_wheel_momentum(&mut y, &runtime).unwrap();
        assert!((y[13] - 0.5).abs() < 1e-12);

        y[13] = -0.8;
        project_wheel_momentum(&mut y, &runtime).unwrap();
        assert!((y[13] + 0.5).abs() < 1e-12);
    }

    #[test]
    fn derivative_and_integrators_support_reaction_wheels() {
        let runtime = test_runtime_with_wheel();
        let y = vec![
            1.0, 0.0, 0.0, 0.0, 0.01, -0.02, 0.03, 7_000_000.0, 0.0, 0.0, 0.0, 7_500.0, 0.0, 0.1,
        ];
        let dy = derivative_vec(&y, 2460000.5, &runtime).unwrap();
        assert_eq!(dy.len(), 14);
        assert!(dy.iter().all(|v| v.is_finite()));

        let (_, rk4_states) =
            integrate_rk4_native(y.clone(), 2460000.5, 0.5, 1.0, &runtime, false, 10, 1, || {
                Ok::<(), ()>(())
            })
            .unwrap();
        assert_eq!(rk4_states.len(), 3);
        assert_eq!(rk4_states[2].len(), 14);

        let (_, dop_states) = integrate_dop853_native(
            y,
            2460000.5,
            0.5,
            1.0,
            &runtime,
            1e-9,
            vec![1e-12],
            10.0,
            1e-6,
            false,
            10,
            0,
            || Ok::<(), ()>(()),
        )
        .unwrap();
        assert!(!dop_states.is_empty());
        assert_eq!(dop_states.last().unwrap().len(), 14);
    }
}
