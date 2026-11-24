use std::fs;
use std::path::Path;

const J2000: f64 = 2_451_545.0;
const SECONDS_PER_DAY: f64 = 86_400.0;
const TABLE_TARGETS: [i32; 12] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 301, 10, 399];
const BODY_NAMES: [&str; 13] = [
    "earth-moon barycenter", "emb", "jupiter", "mars", "mercury", "moon", "neptune", "pluto",
    "saturn", "sun", "uranus", "venus", "earth",
];
#[derive(Clone)]
pub(crate) struct Segment {
    target: i32,
    center: i32,
    kind: i32,
    offset: usize,
    size: usize,
    start: f64,
    end: f64,
}

pub(crate) struct Kernel {
    pub(crate) segments: Vec<Segment>,
    pub(crate) data: Vec<f64>,
}

#[derive(Clone, Copy)]
pub(crate) struct State {
    pub(crate) position: [f64; 3],
    pub(crate) velocity: [f64; 3],
}

impl Kernel {
    pub(crate) fn load(path: &Path) -> Result<Self, String> {
        let raw = fs::read(path).map_err(|e| format!("cannot open DE421 kernel {}: {e}", path.display()))?;
        if raw.len() < 1024 || raw.get(..8) != Some(b"DAF/SPK ") {
            return Err(String::from("invalid DAF/SPK kernel"));
        }
        let marker = raw.get(88..96).ok_or_else(|| String::from("truncated DAF/SPK header"))?;
        let little = match marker {
            b"LTL-IEEE" => true,
            b"BIG-IEEE" => false,
            _ => return Err(String::from("unsupported DAF byte order")),
        };
        let nd = read_i32(&raw, 8, little)?;
        let ni = read_i32(&raw, 12, little)?;
        if (nd, ni) != (2, 6) {
            return Err(String::from("unsupported DAF summary format"));
        }
        let mut record = read_i32(&raw, 76, little)?;
        let summary_size = (nd * 8 + ni * 4) as usize;
        let mut segments = Vec::new();
        let mut data = Vec::new();
        while record != 0 {
            if record < 1 {
                return Err(String::from("invalid DAF summary record address"));
            }
            let base = (record as usize - 1) * 1024;
            if base + 1024 > raw.len() {
                return Err(String::from("invalid DAF summary record"));
            }
            let next = read_f64(&raw, base, little)? as i32;
            let count_value = read_f64(&raw, base + 16, little)?;
            if !(0.0..=25.0).contains(&count_value) {
                return Err(String::from("invalid DAF summary count"));
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
                    return Err(String::from("invalid SPK data addresses"));
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
            return Err(String::from("no supported type 2/3 SPK segments found"));
        }
        Ok(Self { segments, data })
    }

    fn selected(&self, target: i32, et: f64) -> Result<&Segment, String> {
        self.segments.iter().filter(|s| s.target == target && s.start <= et && et <= s.end)
            .last().ok_or_else(|| String::from("no SPK segment covers the requested target and epoch"))
    }

    fn segment_state(&self, segment: &Segment, et: f64) -> Result<State, String> {
        if segment.size < 4 { return Err(String::from("truncated SPK segment")); }
        let trailer = segment.offset + segment.size - 4;
        let initial = self.data[trailer];
        let interval = self.data[trailer + 1];
        let record_size = self.data[trailer + 2] as usize;
        let record_count = self.data[trailer + 3] as usize;
        if interval <= 0.0 || record_size < 5 || record_count == 0 {
            return Err(String::from("invalid SPK segment directory"));
        }
        let mut rec = ((et - initial) / interval) as isize;
        rec = rec.clamp(0, record_count as isize - 1);
        let base = segment.offset + rec as usize * record_size;
        if base + record_size > trailer { return Err(String::from("SPK record exceeds its segment")); }
        let midpoint = self.data[base];
        let radius = self.data[base + 1];
        if radius == 0.0 { return Err(String::from("invalid zero SPK record radius")); }
        let x = (et - midpoint) / radius;
        let ncf = (record_size - 2) / if segment.kind == 2 { 3 } else { 6 };
        if ncf == 0 { return Err(String::from("invalid SPK Chebyshev record")); }
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

    fn barycentric_state(&self, target: i32, et: f64) -> Result<State, String> {
        let mut sum = State { position: [0.0; 3], velocity: [0.0; 3] };
        let mut node = target;
        for _ in 0..8 {
            if node == 0 { return Ok(sum); }
            let segment = self.selected(node, et)?;
            let state = self.segment_state(segment, et)?;
            for i in 0..3 { sum.position[i] += state.position[i]; sum.velocity[i] += state.velocity[i]; }
            node = segment.center;
        }
        Err(String::from("SPK center chain is too deep"))
    }

    pub(crate) fn planetary_state(&self, body_index: usize, jd: f64) -> Result<State, String> {
        if body_index >= TABLE_TARGETS.len() { return Err(String::from("body_index must be in the DE421 body range 0..11")); }
        let target = TABLE_TARGETS[body_index];
        if target == 399 { return Ok(State { position: [0.0; 3], velocity: [0.0; 3] }); }
        let et = (jd - J2000) * SECONDS_PER_DAY;
        let t = self.barycentric_state(target, et)?;
        let e = self.barycentric_state(399, et)?;
        Ok(State { position: sub(t.position, e.position), velocity: sub(t.velocity, e.velocity) })
    }
}
fn read_i32(bytes: &[u8], offset: usize, little: bool) -> Result<i32, String> {
    let b: [u8; 4] = bytes.get(offset..offset + 4).ok_or_else(|| String::from("truncated SPK integer"))?.try_into().unwrap();
    Ok(if little { i32::from_le_bytes(b) } else { i32::from_be_bytes(b) })
}
fn read_f64(bytes: &[u8], offset: usize, little: bool) -> Result<f64, String> {
    let b: [u8; 8] = bytes.get(offset..offset + 8).ok_or_else(|| String::from("truncated SPK double"))?.try_into().unwrap();
    Ok(if little { f64::from_le_bytes(b) } else { f64::from_be_bytes(b) })
}
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
fn scale(a: [f64; 3], s: f64) -> [f64; 3] { [a[0]*s,a[1]*s,a[2]*s] }
pub(crate) fn eclipse_calc(r: [f64; 3], sun: [f64; 3], body_radius: f64, sun_radius: f64) -> (f64, bool, bool) {
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
