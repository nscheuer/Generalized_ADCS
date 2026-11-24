pub fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

pub fn vector_normalize(values: &[f64]) -> Vec<f64> {
    let norm = vector_norm(values);
    if norm == 0.0 {
        vec![0.0; values.len()]
    } else {
        values.iter().map(|value| value / norm).collect()
    }
}

pub fn vector_unit(values: &[f64]) -> Result<Vec<f64>, &'static str> {
    let norm = vector_norm(values);
    if norm == 0.0 {
        Err("cannot normalize a zero vector")
    } else {
        Ok(values.iter().map(|value| value / norm).collect())
    }
}

pub fn vector_skew([x, y, z]: [f64; 3]) -> [[f64; 3]; 3] {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}
