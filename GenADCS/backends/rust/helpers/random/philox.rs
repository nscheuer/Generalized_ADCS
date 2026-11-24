//! Counter-based Philox4x32-10 generator and fixed Box-Muller normal mapping.

const MASK32: u64 = 0xffff_ffff;
const INVERSE_U32_RANGE: f64 = 1.0 / 4_294_967_296.0;
const TWO_PI: f64 = 6.283_185_307_179_586_476_925_286_766_559;

/// Generate the four output words for one Philox counter block.
///
/// Counter words are tick low, tick high, stream identifier, and block index.
/// The 64-bit seed provides the two key words.
pub fn philox4x32(seed: u64, tick: u64, sensor: u32, block: u32) -> [u32; 4] {
    let mut c0 = tick & MASK32;
    let mut c1 = (tick >> 32) & MASK32;
    let mut c2 = u64::from(sensor);
    let mut c3 = u64::from(block);
    let mut k0 = seed & MASK32;
    let mut k1 = (seed >> 32) & MASK32;

    for _ in 0..10 {
        let p0 = 0xD251_1F53_u64 * c0;
        let p1 = 0xCD9E_8D57_u64 * c2;
        let next_c0 = ((p1 >> 32) ^ c1 ^ k0) & MASK32;
        let next_c1 = p1 & MASK32;
        let next_c2 = ((p0 >> 32) ^ c3 ^ k1) & MASK32;
        let next_c3 = p0 & MASK32;
        c0 = next_c0;
        c1 = next_c1;
        c2 = next_c2;
        c3 = next_c3;
        k0 = k0.wrapping_add(0x9E37_79B9) & MASK32;
        k1 = k1.wrapping_add(0xBB67_AE85) & MASK32;
    }

    [c0 as u32, c1 as u32, c2 as u32, c3 as u32]
}

/// Generate standard normal variates with the same fixed Box-Muller mapping
/// used by the companion implementations.
pub fn standard_normals(seed: u64, tick: u64, sensor: u32, count: usize) -> Vec<f64> {
    let mut output = vec![0.0; count];
    let blocks = count.saturating_add(3) / 4;
    for block in 0..blocks {
        let words = philox4x32(seed, tick, sensor, block as u32);
        for pair in 0..2 {
            let offset = block * 4 + pair * 2;
            if offset >= count {
                break;
            }
            let u1 = (f64::from(words[pair * 2]) + 0.5) * INVERSE_U32_RANGE;
            let u2 = (f64::from(words[pair * 2 + 1]) + 0.5) * INVERSE_U32_RANGE;
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = TWO_PI * u2;
            output[offset] = radius * angle.cos();
            if offset + 1 < count {
                output[offset + 1] = radius * angle.sin();
            }
        }
    }
    output
}
