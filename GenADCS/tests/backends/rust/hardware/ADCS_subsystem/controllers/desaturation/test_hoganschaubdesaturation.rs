use crate::controllers::desaturation::{
    hogan_schaub_u0, Desaturation, HoganSchaubManagement, NoDesaturation,
};

#[test]
fn test_hogan_schaub_validation_and_u0() {
    assert!(HoganSchaubManagement::new(0.0, 0.0, 0.0).is_err());
    assert!(HoganSchaubManagement::new(0.5, f64::NAN, 0.0).is_err());

    let hs = HoganSchaubManagement::new(0.5, 0.1, 0.2).unwrap();
    let p = hs.to_params();
    assert!((p[0] - 0.5).abs() < 1e-12);
    assert!((p[1] - 0.1).abs() < 1e-12);
    assert!((p[2] - 0.2).abs() < 1e-12);

    let p0 = NoDesaturation.to_params();
    assert_eq!(p0, vec![0.0, 0.0, 0.0]);

    let u0 = hogan_schaub_u0(&[0.3, -0.1, 0.1], 0.5, 0.1);
    assert!((u0[0] - 0.1).abs() < 1e-12);
    assert!((u0[1] + 0.1).abs() < 1e-12);
    assert!(u0[2].abs() < 1e-12);
}
