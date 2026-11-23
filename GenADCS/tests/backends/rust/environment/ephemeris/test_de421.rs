use super::*;

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn bundled_kernel_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../kernels/environment/ephemeris/de421/de421.bsp")
}

#[test]
fn packaged_de421_kernel_loads_and_returns_plausible_sun_state() {
    let kernel = Kernel::load(&bundled_kernel_path()).unwrap();
    let state = kernel
        .planetary_state(body_name_index("sun").unwrap(), 2_458_849.5)
        .unwrap();
    let position = dot(state.position, state.position).sqrt();
    let velocity = dot(state.velocity, state.velocity).sqrt();
    assert!(
        1.4e8 < position && position < 1.6e8,
        "sun distance={position}"
    );
    assert!(
        25.0 < velocity && velocity < 35.0,
        "sun velocity={velocity}"
    );
    let earth = kernel
        .planetary_state(body_name_index("earth").unwrap(), 2_458_849.5)
        .unwrap();
    assert_eq!(earth.position, [0.0; 3]);
    assert_eq!(earth.velocity, [0.0; 3]);
}

#[test]
fn de421_python_bindings_match_scalar_vector_and_state_contracts() {
    Python::with_gil(|py| {
        let module = PyModule::new(py, "_ephemeris").unwrap();
        _ephemeris(py, module).unwrap();
        let model = module.getattr("DE421").unwrap().call0().unwrap();
        let bodies: Vec<String> = model
            .getattr("available_bodies")
            .unwrap()
            .extract()
            .unwrap();
        assert!(bodies.iter().any(|name| name == "earth"));
        assert!(bodies.iter().any(|name| name == "sun"));

        let epoch = 2_458_849.5;
        let position = model.call_method1("position_eci", ("sun", epoch)).unwrap();
        let velocity = model.call_method1("velocity_eci", ("sun", epoch)).unwrap();
        let state = model.call_method1("state_eci", ("sun", epoch)).unwrap();
        let pos: Vec<f64> = position.call_method0("tolist").unwrap().extract().unwrap();
        let vel: Vec<f64> = velocity.call_method0("tolist").unwrap().extract().unwrap();
        let state_pos: Vec<f64> = state
            .get_item(0)
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        let state_vel: Vec<f64> = state
            .get_item(1)
            .unwrap()
            .call_method0("tolist")
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(pos.len(), 3);
        assert_eq!(vel.len(), 3);
        for i in 0..3 {
            assert!((pos[i] - state_pos[i]).abs() < 1e-8);
            assert!((vel[i] - state_vel[i]).abs() < 1e-12);
        }

        let positions = model
            .call_method1("position_eci", ("moon", vec![epoch, epoch + 1.0]))
            .unwrap();
        let shape: Vec<usize> = positions.getattr("shape").unwrap().extract().unwrap();
        assert_eq!(shape, vec![2, 3]);
        let values: Vec<Vec<f64>> = positions.call_method0("tolist").unwrap().extract().unwrap();
        assert!(values.iter().flatten().all(|value| value.is_finite()));
        assert_ne!(
            norm([values[0][0], values[0][1], values[0][2]]),
            norm([values[1][0], values[1][1], values[1][2]])
        );
    });
}

#[test]
fn de421_resolves_names_and_ids_and_rejects_unknown_bodies() {
    assert_eq!(body_name_index("SUN").unwrap(), 10);
    assert_eq!(body_index_value(301), 9);
    assert!(body_name_index("not-a-body").is_err());
    Python::with_gil(|py| {
        let unknown = 123456i32.into_py(py);
        assert!(body_index(unknown.as_ref(py)).is_err());
    });
}

fn body_index_value(id: i32) -> usize {
    Python::with_gil(|py| {
        let value = id.into_py(py);
        body_index(value.as_ref(py)).unwrap()
    })
}

#[test]
fn de421_rejects_out_of_coverage_epochs_and_malformed_kernel() {
    let kernel = Kernel::load(&bundled_kernel_path()).unwrap();
    assert!(kernel
        .planetary_state(body_name_index("sun").unwrap(), 1_000_000.0)
        .is_err());

    let path = std::env::temp_dir().join(format!("bad-de421-{}.bsp", std::process::id()));
    fs::write(&path, b"not an SPK").unwrap();
    assert!(Kernel::load(&path).is_err());
    fs::remove_file(path).unwrap();
}
