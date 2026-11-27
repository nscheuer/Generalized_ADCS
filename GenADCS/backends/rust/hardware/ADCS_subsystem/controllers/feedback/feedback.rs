use nalgebra::{Matrix3, SymmetricEigen};

pub const FB_MRP_FEEDBACK_LINEARIZED: i32 = 1;

pub type Mat3 = [[f64; 3]; 3];

#[derive(Clone, Copy, Debug)]
pub enum GainInput {
    Scalar(f64),
    Vector([f64; 3]),
    Matrix(Mat3),
}

impl From<f64> for GainInput {
    fn from(value: f64) -> Self {
        Self::Scalar(value)
    }
}

impl From<[f64; 3]> for GainInput {
    fn from(value: [f64; 3]) -> Self {
        Self::Vector(value)
    }
}

impl From<Mat3> for GainInput {
    fn from(value: Mat3) -> Self {
        Self::Matrix(value)
    }
}

pub fn to_gain_matrix(value: GainInput, name: &str, require_spd: bool) -> Result<Mat3, String> {
    let mat = match value {
        GainInput::Scalar(s) => [[s, 0.0, 0.0], [0.0, s, 0.0], [0.0, 0.0, s]],
        GainInput::Vector(v) => [[v[0], 0.0, 0.0], [0.0, v[1], 0.0], [0.0, 0.0, v[2]]],
        GainInput::Matrix(m) => m,
    };

    for row in &mat {
        for &val in row {
            if !val.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
    }

    for i in 0..3 {
        for j in (i + 1)..3 {
            let diff = (mat[i][j] - mat[j][i]).abs();
            let tol = 1e-8 + 1e-5 * mat[j][i].abs();
            if diff > tol {
                return Err(format!("{name} must be symmetric"));
            }
        }
    }

    let sym = Matrix3::new(
        mat[0][0],
        0.5 * (mat[0][1] + mat[1][0]),
        0.5 * (mat[0][2] + mat[2][0]),
        0.5 * (mat[0][1] + mat[1][0]),
        mat[1][1],
        0.5 * (mat[1][2] + mat[2][1]),
        0.5 * (mat[0][2] + mat[2][0]),
        0.5 * (mat[1][2] + mat[2][1]),
        mat[2][2],
    );
    let eig = SymmetricEigen::new(sym);
    let min_eig = eig
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if require_spd && min_eig <= 0.0 {
        return Err(format!("{name} must be positive definite"));
    }
    if !require_spd && min_eig < 0.0 {
        return Err(format!("{name} must be positive semidefinite"));
    }
    Ok(mat)
}

pub trait Feedback: Send + Sync {
    fn kind(&self) -> i32;
    fn to_matrices(&self) -> (Mat3, Mat3);
}
