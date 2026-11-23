use pyo3::exceptions::{PyIOError, PyValueError};
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use once_cell::sync::OnceCell;
use std::sync::Arc;

const J2000: f64 = 2_451_545.0;
const SECONDS_PER_DAY: f64 = 86_400.0;
const TABLE_TARGETS: [i32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 301, 10, 399];
const BODY_NAMES: [&str; 12] = [
    "earth-moon barycenter", "emb", "jupiter", "mars", "mercury", "moon", "neptune", "pluto",
    "saturn", "sun", "uranus", "venus",
];

fn default_kernel_path(py: Python<'_>) -> PyResult<PathBuf> {
    let package_file: String = py.import("GenADCS")?.getattr("__file__")?.extract()?;
    let package_dir = Path::new(&package_file).parent().ok_or_else(|| {
        PyIOError::new_err("cannot locate GenADCS package directory")
    })?;
    Ok(package_dir.join("kernels/environment/ephemeris/de421/de421.bsp"))
}

static DEFAULT_KERNEL: OnceCell<Arc<Kernel>> = OnceCell::new();

fn bundled_kernel(py: Python<'_>) -> PyResult<Arc<Kernel>> {
    DEFAULT_KERNEL
        .get_or_try_init(|| Kernel::load(&default_kernel_path(py)?).map(Arc::new))
        .map(Arc::clone)
}

#[derive(Clone)]
struct Segment {
    target: i32,
    center: i32,
    kind: i32,
    offset: usize,
    size: usize,
    start: f64,
    end: f64,
}

struct Kernel {
    segments: Vec<Segment>,
    data: Vec<f64>,
}

#[derive(Clone, Copy)]
struct State {
    position: [f64; 3],
    velocity: [f64; 3],
}

impl Kernel {
    fn from_arrays(py: Python<'_>, segments: &PyAny, times: &PyAny, data: &PyAny) -> PyResult<Self> {
        let segment_rows: Vec<Vec<i64>> = py.import("numpy")?.call_method1("asarray", (segments,))?
            .call_method1("astype", ("int64",))?.call_method0("tolist")?.extract()?;
        let time_rows: Vec<Vec<f64>> = py.import("numpy")?.call_method1("asarray", (times,))?
            .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
        let data: Vec<f64> = py.import("numpy")?.call_method1("asarray", (data,))?
            .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
        if segment_rows.len() != time_rows.len() || segment_rows.iter().any(|r| r.len() != 5)
            || time_rows.iter().any(|r| r.len() != 2) || data.iter().any(|x| !x.is_finite()) {
            return Err(PyValueError::new_err("invalid parsed SPK segment tables"));
        }
        let segments = segment_rows.into_iter().zip(time_rows).map(|(s, t)| Segment {
            target: s[0] as i32, center: s[1] as i32, kind: s[2] as i32,
            offset: s[3] as usize, size: s[4] as usize, start: t[0], end: t[1],
        }).collect();
        Ok(Self { segments, data })
    }

    fn load(path: &Path) -> PyResult<Self> {
        let raw = fs::read(path).map_err(|e| PyIOError::new_err(format!("cannot open DE421 kernel {}: {e}", path.display())))?;
        if raw.len() < 1024 || raw.get(..8) != Some(b"DAF/SPK ") {
            return Err(PyValueError::new_err("invalid DAF/SPK kernel"));
        }
        let marker = raw.get(88..96).ok_or_else(|| PyValueError::new_err("truncated DAF/SPK header"))?;
        let little = match marker {
            b"LTL-IEEE" => true,
            b"BIG-IEEE" => false,
            _ => return Err(PyValueError::new_err("unsupported DAF byte order")),
        };
        let nd = read_i32(&raw, 8, little)?;
        let ni = read_i32(&raw, 12, little)?;
        if (nd, ni) != (2, 6) {
            return Err(PyValueError::new_err("unsupported DAF summary format"));
        }
        let mut record = read_i32(&raw, 76, little)?;
        let summary_size = (nd * 8 + ni * 4) as usize;
        let mut segments = Vec::new();
        let mut data = Vec::new();
        while record != 0 {
            if record < 1 {
                return Err(PyValueError::new_err("invalid DAF summary record address"));
            }
            let base = (record as usize - 1) * 1024;
            if base + 1024 > raw.len() {
                return Err(PyValueError::new_err("invalid DAF summary record"));
            }
            let next = read_f64(&raw, base, little)? as i32;
            let count_value = read_f64(&raw, base + 16, little)?;
            if !(0.0..=25.0).contains(&count_value) {
                return Err(PyValueError::new_err("invalid DAF summary count"));
            }
            for i in 0..count_value as usize {
                let off = base + 24 + i * summary_size;
                let start = read_f64(&raw, off, little)?;
                let end = read_f64(&raw, off + 8, little)?;
                let ints = off + 16;
                let target = read_i32(&raw, ints, little)?;
                let center = read_i32(&raw, ints + 4, little)?;
                let kind = read_i32(&raw, ints + 12, little)?;
                let first = read_i32(&raw, ints + 16, little)?;
                let last = read_i32(&raw, ints + 20, little)?;
                if kind != 2 && kind != 3 {
                    continue;
                }
                if first < 1 || last < first || last as usize * 8 > raw.len() {
                    return Err(PyValueError::new_err("invalid SPK data addresses"));
                }
                let offset = data.len();
                let size = (last - first + 1) as usize;
                for word in 0..size {
                    data.push(read_f64(&raw, (first as usize - 1 + word) * 8, little)?);
                }
                segments.push(Segment { target, center, kind, offset, size, start, end });
            }
            record = next;
        }
        if segments.is_empty() {
            return Err(PyValueError::new_err("no supported type 2/3 SPK segments found"));
        }
        Ok(Self { segments, data })
    }

    fn selected(&self, target: i32, et: f64) -> PyResult<&Segment> {
        self.segments.iter().filter(|s| s.target == target && s.start <= et && et <= s.end)
            .last().ok_or_else(|| PyValueError::new_err("no SPK segment covers the requested target and epoch"))
    }

    fn segment_state(&self, segment: &Segment, et: f64) -> PyResult<State> {
        if segment.size < 4 { return Err(PyValueError::new_err("truncated SPK segment")); }
        let trailer = segment.offset + segment.size - 4;
        let initial = self.data[trailer];
        let interval = self.data[trailer + 1];
        let record_size = self.data[trailer + 2] as usize;
        let record_count = self.data[trailer + 3] as usize;
        if interval <= 0.0 || record_size < 5 || record_count == 0 {
            return Err(PyValueError::new_err("invalid SPK segment directory"));
        }
        let mut rec = ((et - initial) / interval) as isize;
        rec = rec.clamp(0, record_count as isize - 1);
        let base = segment.offset + rec as usize * record_size;
        if base + record_size > trailer { return Err(PyValueError::new_err("SPK record exceeds its segment")); }
        let midpoint = self.data[base];
        let radius = self.data[base + 1];
        if radius == 0.0 { return Err(PyValueError::new_err("invalid zero SPK record radius")); }
        let x = (et - midpoint) / radius;
        let ncf = (record_size - 2) / if segment.kind == 2 { 3 } else { 6 };
        if ncf == 0 { return Err(PyValueError::new_err("invalid SPK Chebyshev record")); }
        let mut result = State { position: [0.0; 3], velocity: [0.0; 3] };
        for axis in 0..3 {
            let p = base + 2 + axis * ncf;
            let v = base + 2 + (axis + 3) * ncf;
            let mut t0 = 1.0;
            let mut t1 = x;
            let mut d0 = 0.0;
            let mut d1 = 1.0;
            result.position[axis] = self.data[p];
            if ncf > 1 {
                result.position[axis] += self.data[p + 1] * x;
                if segment.kind == 3 { result.velocity[axis] = self.data[v] + self.data[v + 1] * x; }
            }
            for k in 2..ncf {
                let t2 = 2.0 * x * t1 - t0;
                let d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
                result.position[axis] += self.data[p + k] * t2;
                if segment.kind == 3 { result.velocity[axis] += self.data[v + k] * t2; }
                t0 = t1; t1 = t2; d0 = d1; d1 = d2;
            }
            if segment.kind == 2 {
                let mut deriv = if ncf > 1 { self.data[p + 1] } else { 0.0 };
                t0 = 1.0; t1 = x; d0 = 0.0; d1 = 1.0;
                for k in 2..ncf {
                    let t2 = 2.0 * x * t1 - t0;
                    let d2 = 2.0 * t1 + 2.0 * x * d1 - d0;
                    deriv += self.data[p + k] * d2;
                    t0 = t1; t1 = t2; d0 = d1; d1 = d2;
                }
                result.velocity[axis] = deriv / radius;
            }
        }
        Ok(result)
    }

    fn barycentric_state(&self, target: i32, et: f64) -> PyResult<State> {
        let mut sum = State { position: [0.0; 3], velocity: [0.0; 3] };
        let mut node = target;
        for _ in 0..8 {
            if node == 0 { return Ok(sum); }
            let segment = self.selected(node, et)?;
            let state = self.segment_state(segment, et)?;
            for i in 0..3 { sum.position[i] += state.position[i]; sum.velocity[i] += state.velocity[i]; }
            node = segment.center;
        }
        Err(PyValueError::new_err("SPK center chain is too deep"))
    }

    fn planetary_state(&self, body_index: usize, jd: f64) -> PyResult<State> {
        if body_index >= TABLE_TARGETS.len() { return Err(PyValueError::new_err("body_index must be in the DE421 body range 0..11")); }
        let target = TABLE_TARGETS[body_index];
        if target == 399 { return Ok(State { position: [0.0; 3], velocity: [0.0; 3] }); }
        let et = (jd - J2000) * SECONDS_PER_DAY;
        let t = self.barycentric_state(target, et)?;
        let e = self.barycentric_state(399, et)?;
        Ok(State { position: sub(t.position, e.position), velocity: sub(t.velocity, e.velocity) })
    }
}

fn kernel_from_optional_arrays(
    py: Python<'_>,
    segments: Option<&PyAny>,
    times: Option<&PyAny>,
    data: Option<&PyAny>,
) -> PyResult<Arc<Kernel>> {
    match (segments, times, data) {
        (None, None, None) => bundled_kernel(py),
        (Some(s), Some(t), Some(d)) => Ok(Arc::new(Kernel::from_arrays(py, s, t, d)?)),
        _ => Err(PyValueError::new_err("segments, times, and data must be supplied together")),
    }
}

fn read_i32(bytes: &[u8], offset: usize, little: bool) -> PyResult<i32> {
    let b: [u8; 4] = bytes.get(offset..offset + 4).ok_or_else(|| PyValueError::new_err("truncated SPK integer"))?.try_into().unwrap();
    Ok(if little { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) })
}
fn read_f64(bytes: &[u8], offset: usize, little: bool) -> PyResult<f64> {
    let b: [u8; 8] = bytes.get(offset..offset + 8).ok_or_else(|| PyValueError::new_err("truncated SPK double"))?.try_into().unwrap();
    Ok(if little { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn scale(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s,a[1]*s,a[2]*s] }

fn body_index(body: &PyAny) -> PyResult<usize> {
    if let Ok(name) = body.extract::<String>() {
        return body_name_index(&name);
    }
    let id = body.extract::<i32>()?;
    match id {
        0 => Ok(10), 1 => Ok(11), 2 => Ok(9),
        199 => Ok(0), 299 => Ok(1), 3 => Ok(2),
        4 | 499 => Ok(3), 5 | 599 => Ok(4), 6 | 699 => Ok(5),
        7 | 799 => Ok(6), 8 | 899 => Ok(7), 9 | 999 => Ok(8),
        301 => Ok(9), 10 => Ok(10), 399 => Ok(11),
        _ => Err(PyValueError::new_err(format!("unsupported DE421 body identifier {id}"))),
    }
}

fn body_name_index(body: &str) -> PyResult<usize> {
    let name = body.to_lowercase();
    match name.as_str() {
            "mercury" => Ok(0), "venus" => Ok(1), "earth-moon barycenter" | "emb" => Ok(2),
            "mars" => Ok(3), "jupiter" => Ok(4), "saturn" => Ok(5), "uranus" => Ok(6),
            "neptune" => Ok(7), "pluto" => Ok(8), "moon" => Ok(9), "sun" => Ok(10), "earth" => Ok(11),
            _ => Err(PyValueError::new_err(format!("unsupported DE421 body {name:?}"))),
    }
}

fn vec3(py: Python<'_>, x: &PyAny, name: &str) -> PyResult<[f64; 3]> {
    let values: Vec<f64> = py.import("numpy")?.call_method1("asarray", (x,))?
        .call_method1("astype", ("float64",))?.call_method0("tolist")?.extract()?;
    if values.len() != 3 || values.iter().any(|x| !x.is_finite()) {
        return Err(PyValueError::new_err(format!("{name} must be a finite 3-vector")));
    }
    Ok([values[0], values[1], values[2]])
}

fn eclipse_calc(r: [f64; 3], sun: [f64; 3], body_radius: f64, sun_radius: f64) -> (f64, bool, bool) {
    let s = sub(sun, r); let b = scale(r, -1.0);
    let ds = dot(s, s).sqrt(); let db = dot(b, b).sqrt();
    if ds <= sun_radius || db <= body_radius { return (0.0, true, false); }
    let asun = (sun_radius / ds).clamp(-1.0, 1.0).asin();
    let abody = (body_radius / db).clamp(-1.0, 1.0).asin();
    let sep = (dot(s, b) / (ds * db)).clamp(-1.0, 1.0).acos();
    if sep >= asun + abody { return (1.0, false, false); }
    if abody >= asun && sep <= abody - asun { return (0.0, true, false); }
    let area = if sep <= (abody - asun).abs() {
        std::f64::consts::PI * abody.min(asun).powi(2)
    } else {
        let x = (sep*sep + asun*asun - abody*abody) / (2.0*sep*asun);
        let y = (sep*sep + abody*abody - asun*asun) / (2.0*sep*abody);
        let rad = ((-sep+asun+abody)*(sep+asun-abody)*(sep-asun+abody)*(sep+asun+abody)).max(0.0);
        asun*asun*x.clamp(-1.0,1.0).acos() + abody*abody*y.clamp(-1.0,1.0).acos() - 0.5*rad.sqrt()
    };
    ((1.0 - area / (std::f64::consts::PI*asun*asun)).clamp(0.0,1.0), false, true)
}

#[pyfunction(name = "eclipse_state")]
fn eclipse_state_py(py: Python<'_>, spacecraft_position_m: &PyAny, sun_position_m: &PyAny, occulting_body_radius_m: f64, sun_radius_m: f64) -> PyResult<(f64, bool, bool)> {
    Ok(eclipse_calc(vec3(py, spacecraft_position_m, "spacecraft_position_m")?, vec3(py, sun_position_m, "sun_position_m")?, occulting_body_radius_m, sun_radius_m))
}

#[pyfunction]
#[pyo3(signature = (body_id, epoch_jd, segments=None, times=None, data=None))]
fn ephemeris_position(py: Python<'_>, body_id: i32, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let index = match body_id { 0=>10, 1=>11, 2=>9, _=>return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")) };
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(index, epoch_jd)?.position.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_id, epoch_jd, segments=None, times=None, data=None))]
fn ephemeris_velocity(py: Python<'_>, body_id: i32, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let index = match body_id { 0=>10, 1=>11, 2=>9, _=>return Err(PyValueError::new_err("body_id must be BODY_SUN, BODY_EARTH, or BODY_MOON")) };
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(index, epoch_jd)?.velocity.to_pyarray(py).into_py(py))
}

#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_position(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(body_index, epoch_jd)?.position.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_velocity(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<PyObject> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    Ok(kernel.planetary_state(body_index, epoch_jd)?.velocity.to_pyarray(py).into_py(py))
}
#[pyfunction]
#[pyo3(signature = (body_index, epoch_jd, segments=None, times=None, data=None))]
fn planetary_state(py: Python<'_>, body_index: usize, epoch_jd: f64, segments: Option<&PyAny>, times: Option<&PyAny>, data: Option<&PyAny>) -> PyResult<(PyObject, PyObject)> {
    let kernel = kernel_from_optional_arrays(py, segments, times, data)?;
    let state = kernel.planetary_state(body_index, epoch_jd)?;
    Ok((state.position.to_pyarray(py).into_py(py), state.velocity.to_pyarray(py).into_py(py)))
}

