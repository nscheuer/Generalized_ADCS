use super::*;

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn igrf14_loads_bundled_coefficients_and_returns_realistic_field() {
    let model = NativeMagneticField::igrf14().unwrap();
    let field = model
        .field_inertial_tesla([6_778_137.0, 0.0, 0.0], calendar_jd(2020, 1, 1))
        .unwrap();
    assert!(field.iter().all(|x| x.is_finite()));
    assert!(1e-5 < norm(field) && norm(field) < 7e-5, "field={field:?}");
    for year in [1899, 2031] {
        assert!(model
            .field_inertial_tesla([6_778_137.0, 0.0, 0.0], calendar_jd(year, 1, 1))
            .is_err());
    }
    assert!(model
        .field_inertial_tesla([0.0; 3], calendar_jd(2020, 1, 1))
        .is_err());
}

#[test]
fn chaos8_core_and_static_fields_are_realistic_and_distinct() {
    let position = [6_778_137.0, 0.0, 0.0];
    let epoch = calendar_jd(2020, 1, 1);
    let core = NativeMagneticField::chaos8(false)
        .unwrap()
        .field_inertial_tesla(position, epoch)
        .unwrap();
    let full = NativeMagneticField::chaos8(true)
        .unwrap()
        .field_inertial_tesla(position, epoch)
        .unwrap();
    assert!(core.iter().all(|x| x.is_finite()));
    assert!(1e-5 < norm(core) && norm(core) < 7e-5);
    assert!(full.iter().all(|x| x.is_finite()));
    assert!(norm([full[0] - core[0], full[1] - core[1], full[2] - core[2]]) > 0.0);
    assert!(NativeMagneticField::chaos8(false)
        .unwrap()
        .field_inertial_tesla(position, calendar_jd(1800, 1, 1))
        .is_err());
}

#[test]
fn shc_parser_accepts_bundled_data_and_rejects_malformed_rows() {
    let parsed = parse_shc(
        include_str!("../../../../../kernels/environment/magnetic_field_model/igrf14/IGRF14.shc"),
        1,
        13,
    )
    .unwrap();
    assert!(!parsed.epochs.is_empty());
    for text in [
        "1 2 1 0 0\n2020\n1 0\n",
        "1 1 2 0 0\n2020\n1 0 1\n",
        "1 1 1 0 0\n2020\n1 0\n",
    ] {
        assert!(parse_shc(text, 1, 1).is_err());
    }
}

#[test]
fn wmm_placeholder_reports_unimplemented_model() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_magnetic_field_model").unwrap();
        _magnetic_field_model(py, module).unwrap();
        let model = module.getattr("WMM").unwrap().call0().unwrap();
        let error = model
            .call_method1("field", (vec![6_778_137.0, 0.0, 0.0], 2_458_849.5))
            .unwrap_err();
        assert!(error.to_string().contains("Choose a magnetic field model"));
    });
}
