use super::*;

/// Evaluate a magnetic field from one epoch of Schmidt-normalized coefficients.
pub fn magnetic_field_inertial_tesla(
    position_inertial_m: [f64; 3],
    epoch_jd: f64,
    g: &[Vec<f64>],
    h: &[Vec<f64>],
    nmax: usize,
) -> Result<[f64; 3], String> {
    if position_inertial_m.iter().any(|value| !value.is_finite()) || !epoch_jd.is_finite() {
        return Err("position and epoch must be finite".to_owned());
    }
    if g.is_empty() || h.is_empty() || g.len() <= nmax || h.len() <= nmax {
        return Err("coefficient arrays are too small".to_owned());
    }
    if (0..=nmax).any(|n| g[n].len() <= n || h[n].len() <= n) {
        return Err("coefficient arrays are too small".to_owned());
    }

    let side = nmax + 1;
    let mut g_flat = vec![0.0; side * side];
    let mut h_flat = vec![0.0; side * side];
    for n in 0..=nmax {
        for m in 0..=n {
            g_flat[n * side + m] = g[n][m];
            h_flat[n * side + m] = h[n][m];
        }
    }
    evaluate(position_inertial_m, epoch_jd, &g_flat, &h_flat, nmax)
        .map_err(|error| error.to_string())
}
