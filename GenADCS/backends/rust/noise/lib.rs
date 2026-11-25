#![allow(non_local_definitions)]

use pyo3::prelude::*;

pub mod flicker_noise;
mod flicker_noise_py;
pub mod gauss_markov_noise;
mod gauss_markov_noise_py;
pub mod gaussian_white_noise;
mod gaussian_white_noise_py;
pub mod noise;
mod noise_py;
pub mod quantization_noise;
mod quantization_noise_py;
pub mod uniform_noise;
mod uniform_noise_py;

pub use flicker_noise::FlickerNoise as NativeFlickerNoise;
pub use gauss_markov_noise::GaussMarkovNoise as NativeGaussMarkovNoise;
pub use gaussian_white_noise::GaussianWhiteNoise as NativeGaussianWhiteNoise;
pub use noise::Noise as NativeNoise;
pub use quantization_noise::QuantizationNoise as NativeQuantizationNoise;
pub use uniform_noise::UniformNoise as NativeUniformNoise;

#[cfg(test)]
#[path = "../../../tests/backends/rust/noise/test_noise.rs"]
mod test_noise;

#[pymodule]
fn _noise(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<noise_py::Noise>()?;
    module.add_class::<flicker_noise_py::PyFlickerNoise>()?;
    module.add_class::<gauss_markov_noise_py::PyGaussMarkovNoise>()?;
    module.add_class::<gaussian_white_noise_py::GaussianWhiteNoise>()?;
    module.add_class::<quantization_noise_py::QuantizationNoise>()?;
    module.add_class::<uniform_noise_py::UniformNoise>()?;
    Ok(())
}
