use super::super::Runtime;

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
fn unit_rate(v: [f64; 3], dv: [f64; 3]) -> Result<([f64; 3], [f64; 3]), String> {
    let length = norm(v);
    if !length.is_finite() || length == 0.0 {
        return Err("reference direction is undefined".into());
    }
    let u = scale(v, 1.0 / length);
    Ok((u, scale(sub(dv, scale(u, dot(u, dv))), 1.0 / length)))
}

fn dcm_quaternion(m: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let q = if trace > 0.0 {
        let s = 2.0 * (trace + 1.0).sqrt();
        [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    };
    let n = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    let mut result = q.map(|v| v / n);
    if result[0] < 0.0 {
        result = result.map(|v| -v);
    }
    result
}

pub(crate) fn set_reference(
    y: &mut [f64],
    jd: f64,
    mode: i32,
    runtime: &Runtime,
) -> Result<(), String> {
    let position = [y[7], y[8], y[9]];
    let velocity = [y[10], y[11], y[12]];
    let radius = norm(position);
    if radius == 0.0 {
        return Err("orbital radius must be positive".into());
    }
    let acceleration = scale(position, -runtime.mu / radius.powi(3));
    let (radial, radial_dot) = unit_rate(position, velocity)?;
    let radial_speed = dot(velocity, radial);
    let tangent_raw = sub(velocity, scale(radial, radial_speed));
    let tangent_dot_raw = sub(
        sub(
            acceleration,
            scale(
                radial,
                dot(acceleration, radial) + dot(velocity, radial_dot),
            ),
        ),
        scale(radial_dot, radial_speed),
    );
    let (tangent, tangent_dot) = unit_rate(tangent_raw, tangent_dot_raw)?;

    let (z, z_dot, reference, reference_dot) = match mode {
        0 => (
            scale(radial, -1.0),
            scale(radial_dot, -1.0),
            tangent,
            tangent_dot,
        ),
        1 => (tangent, tangent_dot, radial, radial_dot),
        2 => {
            let tdb = magnetic_native::native_tdb_jd(jd)?;
            let eph = runtime
                .ephemeris
                .as_ref()
                .ok_or("Sun pointing requires an ephemeris")?;
            let sun = eph.position_km(0, tdb)?.map(|v| v * 1000.0);
            let sun_velocity = eph.velocity_km_s(0, tdb)?.map(|v| v * 1000.0);
            let (z, z_dot) = unit_rate(sub(sun, position), sub(sun_velocity, velocity))?;
            let mut reference = tangent;
            let mut reference_dot = tangent_dot;
            let projected = sub(reference, scale(z, dot(reference, z)));
            if norm(projected) < 1e-10 {
                reference = radial;
                reference_dot = radial_dot;
            }
            (z, z_dot, reference, reference_dot)
        }
        _ => {
            return Err(
                "reference mode must be 0 (nadir), 1 (along-track), or 2 (sun-pointing)".into(),
            )
        }
    };
    let scalar = dot(reference, z);
    let scalar_dot = dot(reference_dot, z) + dot(reference, z_dot);
    let x_raw = sub(reference, scale(z, scalar));
    let x_dot_raw = sub(
        sub(reference_dot, scale(z, scalar_dot)),
        scale(z_dot, scalar),
    );
    let (x, x_dot) = unit_rate(x_raw, x_dot_raw)?;
    let y_raw = cross(z, x);
    let y_dot_raw = add(cross(z_dot, x), cross(z, x_dot));
    let (body_y, body_y_dot) = unit_rate(y_raw, y_dot_raw)?;
    let body_x = cross(body_y, z);
    let body_x_dot = add(cross(body_y_dot, z), cross(body_y, z_dot));
    y[..4].copy_from_slice(&dcm_quaternion([body_x, body_y, z]));
    y[4] = -dot(body_y_dot, z);
    y[5] = -dot(z_dot, body_x);
    y[6] = -dot(body_x_dot, body_y);
    Ok(())
}
