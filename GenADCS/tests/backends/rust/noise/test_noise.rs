use super::{
    NativeFlickerNoise as FlickerNoise, NativeGaussMarkovNoise as GaussMarkovNoise,
    NativeGaussianWhiteNoise as GaussianWhiteNoise, NativeNoise as Noise,
    NativeQuantizationNoise as QuantizationNoise, NativeUniformNoise as UniformNoise,
};

#[test]
fn covariance_noise_builds_cholesky_and_applies_gaussian_draw() {
    let noise = Noise::new(Some(vec![vec![4.0, 2.0], vec![2.0, 3.0]])).unwrap();
    assert_eq!(
        noise.cholesky_factor.as_ref().unwrap(),
        &vec![vec![2.0, 0.0], vec![1.0, 2.0_f64.sqrt()]]
    );
    let sample = noise.gaussian_sample(12, 3).unwrap();
    assert_eq!(sample, noise.gaussian_sample(12, 3).unwrap());
    assert_ne!(sample, noise.gaussian_sample(12, 4).unwrap());
    let result = noise.apply_with_draw(&[1.0, -1.0], &[0.5, -0.25]).unwrap();
    assert!((result[0] - 2.0).abs() < 1e-12);
    assert!((result[1] - (-1.0 + 0.5 - 0.25 * 2.0_f64.sqrt())).abs() < 1e-12);
}

#[test]
fn covariance_validation_matches_python_contract() {
    assert!(Noise::new(None).unwrap().apply(&[1.0], 0.1).is_err());
    assert!(Noise::new(Some(vec![vec![1.0, 0.0], vec![0.0]])).is_err());
    assert!(Noise::new(Some(vec![vec![-1.0]])).is_err());
    assert!(Noise::new(Some(vec![vec![1.0, 0.0], vec![0.0, 2.0]]))
        .unwrap()
        .apply(&[1.0], 0.1)
        .is_err());
}

#[test]
fn concrete_noise_models_store_python_dataclass_parameters() {
    assert_eq!(FlickerNoise::new(0.25).amplitude, 0.25);
    assert_eq!(
        GaussMarkovNoise::new(0.5, 12.0),
        GaussMarkovNoise {
            standard_deviation: 0.5,
            correlation_time: 12.0
        }
    );
    assert_eq!(
        GaussianWhiteNoise::new(0.1).unwrap().standard_deviation,
        0.1
    );
    assert_eq!(QuantizationNoise::new(0.01).unwrap().step_size, 0.01);
    assert_eq!(
        UniformNoise::new(-1.0, 1.0).unwrap(),
        UniformNoise {
            lower: -1.0,
            upper: 1.0
        }
    );
}

#[test]
fn bounded_models_reject_invalid_inputs() {
    assert!(QuantizationNoise::new(f64::NAN).is_err());
    assert!(QuantizationNoise::new(-1.0).is_err());
    assert!(UniformNoise::new(1.0, -1.0).is_err());
    assert!(UniformNoise::new(f64::NEG_INFINITY, 1.0).is_err());
}
