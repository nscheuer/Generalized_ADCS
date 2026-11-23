pub enum NativeMagneticField {
    Igrf {
        coefficients: Coefficients,
        epochs_jd: Vec<f64>,
    },
    Chaos {
        core: Coefficients,
        static_field: Option<Coefficients>,
    },
}
pub fn native_rotation(utc_jd: f64) -> PyResult<[[f64; 3]; 3]> {
    let tt = if utc_jd >= 2441317.5 {
        tt_jd(utc_jd)?
    } else {
        utc_jd
    };
    Ok(gcrf_ecef(utc_jd, tt))
}
pub fn native_tdb_jd(utc_jd: f64) -> PyResult<f64> {
    let tt = tt_jd(utc_jd)?;
    let g = (357.53 + 0.9856003 * (tt - 2451545.0)).to_radians();
    Ok(tt + (0.001657 * g.sin() + 0.000022 * (2.0 * g).sin()) / 86400.0)
}
impl NativeMagneticField {
    pub fn igrf14() -> PyResult<Self> {
        let coefficients = parse_shc(include_str!("../../../../kernels/environment/magnetic_field_model/igrf14/IGRF14.shc"), 1, 13)?;
        let epochs_jd = coefficients
            .epochs
            .iter()
            .map(|year| calendar_jd(*year as i32, 1, 1))
            .collect();
        Ok(Self::Igrf {
            coefficients,
            epochs_jd,
        })
    }
    pub fn chaos8(include_static: bool) -> PyResult<Self> {
        let core = parse_shc(include_str!("../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_core.shc"), 1, 20)?;
        let static_field = if include_static {
            Some(parse_shc(
                include_str!("../../../../kernels/environment/magnetic_field_model/chaos8/CHAOS-8.7_static.shc"),
                21,
                185,
            )?)
        } else {
            None
        };
        Ok(Self::Chaos { core, static_field })
    }
    pub fn field_inertial_tesla(&self, position: [f64; 3], jd: f64) -> PyResult<[f64; 3]> {
        match self {
            Self::Igrf {
                coefficients,
                epochs_jd,
            } => {
                if jd < epochs_jd[0] || jd > epochs_jd[epochs_jd.len() - 1] {
                    return Err(PyValueError::new_err(
                        "IGRF-14 coefficients cover 1900-01-01 to 2030-01-01",
                    ));
                }
                let i = epochs_jd
                    .partition_point(|x| *x <= jd)
                    .saturating_sub(1)
                    .min(epochs_jd.len() - 2);
                let w = (jd - epochs_jd[i]) / (epochs_jd[i + 1] - epochs_jd[i]);
                let (mut g, mut h) = padded(coefficients, i, 13);
                let (g1, h1) = padded(coefficients, i + 1, 13);
                for k in 0..g.len() {
                    g[k] += w * (g1[k] - g[k]);
                    h[k] += w * (h1[k] - h[k]);
                }
                evaluate(position, jd, &g, &h, 13)
            }
            Self::Chaos { core, static_field } => {
                let year = 2000. + (jd - 2451544.5) / 365.25;
                if year < core.epochs[0] || year > core.epochs[core.epochs.len() - 1] {
                    return Err(PyValueError::new_err(
                        "Epoch outside CHAOS-8.7 core coefficient range",
                    ));
                }
                let upper = core.epochs.partition_point(|x| *x <= year);
                let start = ((upper.saturating_sub(1)) / 5 * 5).min(core.epochs.len() - 6);
                let nmax = if static_field.is_some() { 185 } else { 20 };
                let side = nmax + 1;
                let (mut g, mut h) = if let Some(stat) = static_field {
                    padded(stat, 0, nmax)
                } else {
                    (vec![0.; side * side], vec![0.; side * side])
                };
                for j in 0..6 {
                    let mut weight = 1.;
                    for k in 0..6 {
                        if j != k {
                            weight *= (year - core.epochs[start + k])
                                / (core.epochs[start + j] - core.epochs[start + k]);
                        }
                    }
                    let (gj, hj) = padded(core, start + j, nmax);
                    for n in 1..=20 {
                        for m in 0..=n {
                            let ix = n * side + m;
                            g[ix] += weight * gj[ix];
                            h[ix] += weight * hj[ix];
                        }
                    }
                }
                evaluate(position, jd, &g, &h, nmax)
            }
        }
    }
}
