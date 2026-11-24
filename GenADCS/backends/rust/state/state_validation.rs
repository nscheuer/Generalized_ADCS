pub fn finite(values: &[f64], name: &str, size: Option<usize>) -> Result<(), String> {
    if size.map_or(false, |expected| values.len() != expected)
        || values.iter().any(|value| !value.is_finite())
    {
        let description = size.map_or_else(
            || "finite 1D array".to_owned(),
            |length| format!("finite {length}-vector"),
        );
        return Err(format!("{name} must be a {description}"));
    }
    Ok(())
}
