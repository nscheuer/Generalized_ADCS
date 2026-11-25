//! Validated initial state shared by attitude estimators.

pub type Mat6 = [[f64; 6]; 6];
pub type Mat3 = [[f64; 3]; 3];

pub const DIAGNOSTIC_FIELDS: [&str; 10] = [
    "estimated_attitude",
    "estimated_rate",
    "estimated_attitude_covariance",
    "estimated_rate_covariance",
    "measurement_innovation",
    "measurement_innovation_covariance",
    "normalized_innovation_squared",
    "normalized_estimation_error_squared",
    "measurement_residual",
    "kalman_gain",
];

#[derive(Clone, Debug, PartialEq)]
pub struct AttitudeEstimator {
    pub quaternion: [f64; 4],
    pub rate: [f64; 3],
    pub covariance: Mat6,
    pub acceleration_psd: Mat3,
    pub record: Vec<String>,
}

impl AttitudeEstimator {
    pub fn new(
        quaternion: [f64; 4],
        rate: [f64; 3],
        covariance: Mat6,
        acceleration_psd: Mat3,
        record: Vec<String>,
    ) -> Result<Self, &'static str> {
        if !quaternion.iter().all(|x| x.is_finite()) {
            return Err("quaternion must be a finite nonzero Hamilton 4-vector");
        }
        let qnorm = norm4(quaternion);
        if qnorm == 0.0 {
            return Err("quaternion must be a finite nonzero Hamilton 4-vector");
        }
        if !rate.iter().all(|x| x.is_finite()) {
            return Err("rate must be a finite 3-vector");
        }
        if !covariance.iter().flatten().all(|x| x.is_finite()) || !symmetric6(covariance) {
            return Err("covariance must be a finite symmetric 6x6 matrix");
        }
        if min_eigenvalue6(covariance) < -1e-12 {
            return Err("covariance must be positive semidefinite");
        }
        if !acceleration_psd.iter().flatten().all(|x| x.is_finite())
            || !symmetric3(acceleration_psd)
        {
            return Err("acceleration_psd must be a finite symmetric 3x3 matrix");
        }
        if min_eigenvalue3(acceleration_psd) < -1e-12 {
            return Err("acceleration_psd must be positive semidefinite");
        }
        if record
            .iter()
            .any(|name| !DIAGNOSTIC_FIELDS.contains(&name.as_str()))
        {
            return Err("unknown estimator diagnostic field");
        }
        Ok(Self {
            quaternion: quaternion.map(|x| x / qnorm),
            rate,
            covariance,
            acceleration_psd,
            record,
        })
    }

    pub fn prepare(&self) -> ([f64; 4], [f64; 3], Mat6, Mat3) {
        (
            self.quaternion,
            self.rate,
            self.covariance,
            self.acceleration_psd,
        )
    }
}

fn norm4(q: [f64; 4]) -> f64 {
    q.iter().map(|x| x * x).sum::<f64>().sqrt()
}
fn symmetric6(a: Mat6) -> bool {
    (0..6).all(|i| (0..6).all(|j| close(a[i][j], a[j][i])))
}
fn symmetric3(a: Mat3) -> bool {
    (0..3).all(|i| (0..3).all(|j| close(a[i][j], a[j][i])))
}
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

fn min_eigenvalue3(a: Mat3) -> f64 {
    min_eigenvalue(a.iter().map(|r| r.to_vec()).collect())
}
fn min_eigenvalue6(a: Mat6) -> f64 {
    min_eigenvalue(a.iter().map(|r| r.to_vec()).collect())
}
fn min_eigenvalue(mut a: Vec<Vec<f64>>) -> f64 {
    let n = a.len();
    for _ in 0..(64 * n * n) {
        let mut p = 0;
        let mut q = 1;
        let mut largest = 0.0_f64;
        for i in 0..n {
            for j in i + 1..n {
                if a[i][j].abs() > largest {
                    largest = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest < 1e-15 {
            break;
        }
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];
        a[p][p] = app - t * apq;
        a[q][q] = aqq + t * apq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;
        for k in 0..n {
            if k != p && k != q {
                let akp = a[k][p];
                let akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[p][k] = a[k][p];
                a[k][q] = s * akp + c * akq;
                a[q][k] = a[k][q];
            }
        }
    }
    (0..n).map(|i| a[i][i]).fold(f64::INFINITY, f64::min)
}
