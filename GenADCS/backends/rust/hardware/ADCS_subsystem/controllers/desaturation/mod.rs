pub mod desaturation;
pub mod hoganschaubdesaturation;

pub use desaturation::{Desaturation, NoDesaturation, DESAT_HOGAN_SCHAUB, DESAT_NONE};
pub use hoganschaubdesaturation::{
    hogan_schaub_mtq_dump, hogan_schaub_u0, HoganSchaubManagement,
};
