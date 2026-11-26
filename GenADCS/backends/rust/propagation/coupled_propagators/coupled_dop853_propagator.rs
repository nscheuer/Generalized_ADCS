pub fn dop853_vec(
    y: &[f64],
    f: &[f64],
    jd: f64,
    h: f64,
    p: &Runtime,
    rtol: f64,
    atol: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64), String> {
    if y.len() != 13 || f.len() != 13 || atol.len() != 13 {
        return Err("DOP853 vectors must contain 13 values".to_owned());
    }
    let mut k = [[0.; 13]; 13];
    k[0].copy_from_slice(f);
    for stage in 1..12 {
        let mut trial = [0.; 13];
        trial.copy_from_slice(y);
        for prev in 0..stage {
            if A[stage][prev] != 0. {
                for i in 0..y.len() {
                    trial[i] += h * A[stage][prev] * k[prev][i];
                }
            }
        }
        k[stage] = derivative_array(&trial, jd + C[stage] * h / 86400., p)?;
    }
    let mut next = y.to_vec();
    for stage in 0..12 {
        if B[stage] != 0. {
            for i in 0..y.len() {
                next[i] += h * B[stage] * k[stage][i];
            }
        }
    }
    normalize_q(&mut next)?;
    k[12] = derivative_array(&next, jd + h / 86400., p)?;
    let (mut n5, mut n3) = (0., 0.);
    for j in 0..y.len() {
        let scale = atol[j] + rtol * y[j].abs().max(next[j].abs());
        let (mut e5, mut e3) = (0., 0.);
        for stage in 0..13 {
            e5 += E5[stage] * k[stage][j];
            e3 += E3[stage] * k[stage][j];
        }
        n5 += (e5 / scale).powi(2);
        n3 += (e3 / scale).powi(2);
    }
    let error = if n5 == 0. && n3 == 0. {
        0.
    } else {
        h.abs() * n5 / ((n5 + 0.01 * n3) * y.len() as f64).sqrt()
    };
    Ok((next, k[12].to_vec(), error))
}
