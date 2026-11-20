use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

pub fn finite(values: &[f64], name: &str, size: Option<usize>) -> PyResult<()> {
    if size.map_or(false, |expected| values.len() != expected)
        || values.iter().any(|v| !v.is_finite())
    {
        let description = size.map_or_else(
            || "finite 1D array".to_string(),
            |n| format!("finite {n}-vector"),
        );
        return Err(PyValueError::new_err(format!(
            "{name} must be a {description}"
        )));
    }
    Ok(())
}
