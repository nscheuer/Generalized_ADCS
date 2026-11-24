//! Rust-native coordinate-frame selection and conversion operations.

use super::{ecef_geo, gcrf_teme, lvlh, mm, qmat, teme_ecef, tr, M, V};

/// Supported coordinate frames for native vector conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    Gcrf,
    Ecef,
    Body,
    Lvlh,
    Teme,
}

impl Frame {
    /// Parse a frame name, case-insensitively.
    pub fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_uppercase().as_str() {
            "GCRF" => Ok(Self::Gcrf),
            "ECEF" => Ok(Self::Ecef),
            "BODY" => Ok(Self::Body),
            "LVLH" => Ok(Self::Lvlh),
            "TEME" => Ok(Self::Teme),
            _ => Err(format!(
                "unsupported frame {name}; expected GCRF, ECEF, BODY, LVLH, or TEME"
            )),
        }
    }
}

fn from_gcrf(
    frame: Frame,
    jd_ut1: Option<f64>,
    jd_tt: Option<f64>,
    xp: f64,
    yp: f64,
    position_gcrf: Option<V>,
    velocity_gcrf: Option<V>,
    q_body_from_gcrf: Option<[f64; 4]>,
) -> Result<M, String> {
    match frame {
        Frame::Gcrf => Ok(M::identity()),
        Frame::Ecef => {
            let ut1 = jd_ut1.ok_or("jd_ut1 is required for ECEF or TEME conversions")?;
            let tt = jd_tt.ok_or("jd_tt is required for ECEF or TEME conversions")?;
            Ok(mm(teme_ecef(ut1, xp, yp), gcrf_teme(tt)))
        }
        Frame::Teme => {
            let _ut1 = jd_ut1.ok_or("jd_ut1 is required for ECEF or TEME conversions")?;
            let tt = jd_tt.ok_or("jd_tt is required for ECEF or TEME conversions")?;
            Ok(gcrf_teme(tt))
        }
        Frame::Lvlh => {
            let position = position_gcrf
                .ok_or("position_gcrf and velocity_gcrf are required for LVLH conversions")?;
            let velocity = velocity_gcrf
                .ok_or("position_gcrf and velocity_gcrf are required for LVLH conversions")?;
            lvlh(position, velocity).map_err(str::to_owned)
        }
        Frame::Body => Ok(qmat(
            q_body_from_gcrf.ok_or("q_body_from_gcrf is required for body-frame conversions")?,
        )),
    }
}

/// Return the rotation matrix that maps vectors from `source` into `target`.
pub fn context_rotation(
    source: Frame,
    target: Frame,
    jd_ut1: Option<f64>,
    jd_tt: Option<f64>,
    xp: f64,
    yp: f64,
    position_gcrf: Option<V>,
    velocity_gcrf: Option<V>,
    q_body_from_gcrf: Option<[f64; 4]>,
) -> Result<M, String> {
    if source == target {
        return Ok(M::identity());
    }
    let target_from_gcrf = from_gcrf(
        target,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position_gcrf,
        velocity_gcrf,
        q_body_from_gcrf,
    )?;
    let source_from_gcrf = from_gcrf(
        source,
        jd_ut1,
        jd_tt,
        xp,
        yp,
        position_gcrf,
        velocity_gcrf,
        q_body_from_gcrf,
    )?;
    Ok(mm(target_from_gcrf, tr(source_from_gcrf)))
}

/// Convert an ECEF Cartesian position in metres to geodetic degrees and km.
pub fn ecef_to_geodetic(position_ecef_m: V) -> Result<V, String> {
    if position_ecef_m.iter().any(|value| !value.is_finite()) {
        return Err("position_ecef_m must contain only finite values".to_owned());
    }
    Ok(ecef_geo(position_ecef_m))
}

/// Convert a GCRF Cartesian position in metres to geodetic degrees and km.
pub fn gcrf_position_to_geodetic(
    position_gcrf_m: V,
    jd_ut1: f64,
    jd_tt: f64,
    xp: f64,
    yp: f64,
) -> Result<V, String> {
    if position_gcrf_m.iter().any(|value| !value.is_finite()) {
        return Err("position_gcrf_m must contain only finite values".to_owned());
    }
    let ecef = mm(teme_ecef(jd_ut1, xp, yp), gcrf_teme(jd_tt))
        * nalgebra::Vector3::from_row_slice(&position_gcrf_m);
    ecef_to_geodetic([ecef[0], ecef[1], ecef[2]])
}
