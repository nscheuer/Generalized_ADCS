pub fn inertial_to_body_matrix_native(q: [f64; 4]) -> M { qmat(q) }
pub fn body_to_inertial_matrix_native(q: [f64; 4]) -> M { tr(qmat(q)) }
pub fn inertial_to_body_native(v: V, q: [f64; 4]) -> V { mv(qmat(q), v) }
pub fn body_to_inertial_native(v: V, q: [f64; 4]) -> V { mv(tr(qmat(q)), v) }
pub fn transform_vector_native(v: V, m: M) -> V { mv(m, v) }
pub fn transform_covariance_native(c: M, m: M) -> M { mm(mm(m, c), tr(m)) }
pub fn gcrf_to_lvlh_matrix_native(p: V, v: V) -> Result<M, &'static str> { lvlh(p, v) }
pub fn gcrf_to_ecef_matrix_native(jd_ut1: f64, jd_tt: f64, xp: f64, yp: f64) -> M {
    mm(teme_ecef(jd_ut1, xp, yp), gcrf_teme(jd_tt))
}
pub fn ecef_to_gcrf_matrix_native(jd_ut1: f64, jd_tt: f64, xp: f64, yp: f64) -> M {
    tr(gcrf_to_ecef_matrix_native(jd_ut1, jd_tt, xp, yp))
}
pub fn gcrf_to_teme_matrix_native(jd_tt: f64) -> M { gcrf_teme(jd_tt) }
pub fn teme_to_gcrf_matrix_native(jd_tt: f64) -> M { tr(gcrf_teme(jd_tt)) }
pub fn teme_to_ecef_matrix_native(jd_ut1: f64, xp: f64, yp: f64) -> M { teme_ecef(jd_ut1, xp, yp) }
pub fn ecef_to_teme_matrix_native(jd_ut1: f64, xp: f64, yp: f64) -> M { tr(teme_ecef(jd_ut1, xp, yp)) }
