use super::*;

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[test]
fn j2_acceleration_is_realistic_central_and_symmetric() {
    let p = [7_000_000.0, 0.0, 0.0];
    let acceleration = native_acceleration(
        p,
        EARTH_MU_M3_S2,
        EARTH_REFERENCE_RADIUS_M,
        &EARTH_ZONAL_COEFFICIENTS,
    )
    .unwrap();
    assert!(7.5 < norm(acceleration) && norm(acceleration) < 8.5);
    assert!(acceleration[0] < 0.0);
    assert!(acceleration[1].abs() < 1e-14);
    let rotated = native_acceleration(
        [0.0, 7_000_000.0, 0.0],
        EARTH_MU_M3_S2,
        EARTH_REFERENCE_RADIUS_M,
        &EARTH_ZONAL_COEFFICIENTS,
    )
    .unwrap();
    assert!(rotated[0].abs() < 1e-14);
}

#[test]
fn each_zonal_degree_is_finite_and_perturbative() {
    let p = [7_000_000.0, 0.0, 1_000_000.0];
    for degree in 2..=6 {
        let full = JGravityModel::configured(vec![degree], None).unwrap();
        let point_mass = JGravityModel::configured(vec![], None).unwrap();
        let a = acceleration(
            p,
            EARTH_MU_M3_S2,
            EARTH_REFERENCE_RADIUS_M,
            &full.zonal_coefficients,
        )
        .unwrap();
        let b = acceleration(
            p,
            EARTH_MU_M3_S2,
            EARTH_REFERENCE_RADIUS_M,
            &point_mass.zonal_coefficients,
        )
        .unwrap();
        assert!(a.iter().all(|x| x.is_finite()));
        let delta = norm([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        assert!(
            delta > 0.0 && delta < 2e-2,
            "degree {degree} perturbation={delta}"
        );
    }
}

#[test]
fn gravity_rejects_invalid_degrees_short_coefficients_and_centre() {
    assert!(JGravityModel::configured(vec![1], None).is_err());
    assert!(JGravityModel::configured(vec![7], None).is_err());
    assert!(JGravityModel::configured(vec![2], Some(vec![1.0; 4])).is_err());
    assert!(native_acceleration([0.0; 3], EARTH_MU_M3_S2, EARTH_REFERENCE_RADIUS_M, &[]).is_err());
}
