fn vec3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<V> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("shape")?.extract::<(usize,)>().ok() != Some((3,)) {
        return Err(PyValueError::new_err(format!(
            "{name} has an invalid shape"
        )));
    }
    let v: V = a
        .call_method0("tolist")?
        .extract::<Vec<f64>>()?
        .try_into()
        .unwrap();
    if v.iter().any(|n| !n.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(v)
}
fn matrix(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<M> {
    let a = py
        .import("numpy")?
        .call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?;
    if a.getattr("shape")?.extract::<(usize, usize)>().ok() != Some((3, 3)) {
        return Err(PyValueError::new_err(format!(
            "{name} has an invalid shape"
        )));
    }
    let r: Vec<Vec<f64>> = a.call_method0("tolist")?.extract()?;
    let flat: Vec<f64> = r.into_iter().flatten().collect();
    if flat.iter().any(|v| !v.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(M::from_row_slice(&flat))
}
fn pyarr(py: Python<'_>, x: impl IntoPy<PyObject>) -> PyResult<PyObject> {
    Ok(py.import("numpy")?.call_method1("asarray", (x,))?.into())
}
fn pymat(py: Python<'_>, x: &M) -> PyObject {
    x.to_pyarray(py).into_py(py)
}
fn cross(a: V, b: V) -> V {
    let value = SVector::<f64, 3>::from_row_slice(&a).cross(&SVector::from_row_slice(&b));
    [value[0], value[1], value[2]]
}
fn dot(a: V, b: V) -> f64 {
    SVector::<f64, 3>::from_row_slice(&a).dot(&SVector::from_row_slice(&b))
}
fn scale(a: V, s: f64) -> V {
    (SVector::<f64, 3>::from_row_slice(&a) * s).into()
}
fn mm(a: M, b: M) -> M {
    a * b
}
fn mv(a: M, b: V) -> V {
    let out = a * SVector::<f64, 3>::from_row_slice(&b);
    [out[0], out[1], out[2]]
}
fn tr(a: M) -> M {
    a.transpose()
}
fn qmat(q: [f64; 4]) -> M {
    let [w, x, y, z] = q;
    M::from_row_slice(&[
        w * w + x * x - y * y - z * z,
        2. * (x * y - w * z),
        2. * (x * z + w * y),
        2. * (x * y + w * z),
        w * w - x * x + y * y - z * z,
        2. * (y * z - w * x),
        2. * (x * z - w * y),
        2. * (y * z + w * x),
        w * w - x * x - y * y + z * z,
    ])
}
fn r1(a: f64) -> M {
    let (c, s) = (a.cos(), a.sin());
    M::from_row_slice(&[1., 0., 0., 0., c, -s, 0., s, c])
}
fn r2(a: f64) -> M {
    let (c, s) = (a.cos(), a.sin());
    M::from_row_slice(&[c, 0., s, 0., 1., 0., -s, 0., c])
}
fn r3(a: f64) -> M {
    let (c, s) = (a.cos(), a.sin());
    M::from_row_slice(&[c, -s, 0., s, c, 0., 0., 0., 1.])
}
fn precession(jd: f64) -> M {
    let t = (jd - 2451545.) / 36525.;
    let zeta = (2306.2181 * t + 0.30188 * t * t + 0.017998 * t * t * t) * PI / (180. * 3600.);
    let z = (2306.2181 * t + 1.09468 * t * t + 0.018203 * t * t * t) * PI / (180. * 3600.);
    let th = (2004.3109 * t - 0.42665 * t * t - 0.041833 * t * t * t) * PI / (180. * 3600.);
    mm(mm(r3(-z), r2(th)), r3(-zeta))
}
fn gcrf_teme(jd: f64) -> M {
    let t = (jd - 2451545.) / 36525.;
    let om = (125.04452 - 1934.136261 * t).to_radians();
    let ms = (280.4665 + 36000.7698 * t).to_radians();
    let mmn = (218.3165 + 481267.8813 * t).to_radians();
    let dpsi = (-17.20 * om.sin() - 1.32 * (2. * ms).sin() - 0.23 * (2. * mmn).sin()
        + 0.21 * (2. * om).sin())
        * PI
        / (180. * 3600.);
    let deps = (9.20 * om.cos() + 0.57 * (2. * ms).cos() + 0.10 * (2. * mmn).cos()
        - 0.09 * (2. * om).cos())
        * PI
        / (180. * 3600.);
    let eps = (23.43929111 - 0.013004167 * t - 1.64e-7 * t * t + 5.04e-7 * t * t * t).to_radians();
    let nut = mm(mm(r1(-(eps + deps)), r3(-dpsi)), r1(eps));
    mm(mm(r3(-dpsi * eps.cos()), nut), precession(jd))
}
fn gmst(jd: f64) -> f64 {
    let t = (jd - 2451545.) / 36525.;
    let sec = 67310.54841 + ((876600. * 3600.) + 8640184.812866) * t + 0.093104 * t * t
        - 6.2e-6 * t * t * t;
    (sec / 86400. - (sec / 86400.).floor()) * 2. * PI
}
fn teme_ecef(jd: f64, xp: f64, yp: f64) -> M {
    mm(mm(r2(-xp), r1(-yp)), r3(-gmst(jd)))
}
fn lvlh(p: V, v: V) -> PyResult<M> {
    let pn = dot(p, p).sqrt();
    let h = cross(p, v);
    let hn = dot(h, h).sqrt();
    if pn == 0. || hn == 0. {
        return Err(PyValueError::new_err(
            "LVLH requires nonzero position and angular momentum",
        ));
    }
    let x = scale(p, 1. / pn);
    let z = scale(h, 1. / hn);
    let y0 = cross(z, x);
    let yn = dot(y0, y0).sqrt();
    if yn == 0. {
        return Err(PyValueError::new_err("LVLH axes are degenerate"));
    }
    let y = scale(y0, 1. / yn);
    Ok(M::from_row_slice(&[
        x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2],
    ]))
}
fn ecef_geo(p: V) -> V {
    let a = 6378137.;
    let e2: f64 = 6.6943799901413165e-3;
    let rho = p[0].hypot(p[1]);
    let lon = p[1].atan2(p[0]);
    let (lat, alt) = if rho == 0. {
        (
            if p[2] >= 0. { PI / 2. } else { -PI / 2. },
            p[2].abs() - a * (1. - e2).sqrt(),
        )
    } else {
        let mut lat = p[2].atan2(rho * (1. - e2));
        for _ in 0..8 {
            let s = lat.sin();
            let n = a / (1. - e2 * s * s).sqrt();
            lat = (p[2] + e2 * n * s).atan2(rho)
        }
        let s = lat.sin();
        let n = a / (1. - e2 * s * s).sqrt();
        (lat, rho / lat.cos() - n)
    };
    [lat.to_degrees(), lon.to_degrees(), alt / 1000.]
}
