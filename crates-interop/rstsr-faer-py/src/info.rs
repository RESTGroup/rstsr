//! finfo / iinfo result objects. Values come straight from Rust's numeric
//! limits (language constants, not computed algorithms).
//!
//! Note for the gap register: all attributes are Python floats/ints by
//! construction here — the numpy baseline cluster "finfo attrs must be
//! Python floats" cannot occur in this shim.

use pyo3::prelude::*;

use crate::dtype::Dtype;

/// Unified finfo result; constructed by `finfo(dtype)` per float width.
#[pyclass(frozen, module = "rstsr_faer.rstsr_faer")]
pub struct Finfo {
    #[pyo3(get)]
    bits: u32,
    #[pyo3(get)]
    eps: f64,
    #[pyo3(get)]
    max: f64,
    #[pyo3(get)]
    min: f64,
    #[pyo3(get)]
    smallest_normal: f64,
    #[pyo3(get)]
    dtype: Py<Dtype>,
}

#[pymethods]
impl Finfo {
    fn __repr__(&self) -> String {
        format!("finfo(bits={}, eps={:e}, max={:e}, min={:e})", self.bits, self.eps, self.max, self.min)
    }
}

pub fn finfo32(dtype: Py<Dtype>) -> Finfo {
    Finfo {
        bits: 32,
        eps: f32::EPSILON as f64,
        max: f32::MAX as f64,
        min: f32::MIN as f64,
        smallest_normal: f32::MIN_POSITIVE as f64,
        dtype,
    }
}

pub fn finfo64(dtype: Py<Dtype>) -> Finfo {
    Finfo { bits: 64, eps: f64::EPSILON, max: f64::MAX, min: f64::MIN, smallest_normal: f64::MIN_POSITIVE, dtype }
}

/// Unified iinfo result; constructed by `iinfo(dtype)` per integer width.
#[pyclass(frozen, module = "rstsr_faer.rstsr_faer")]
pub struct Iinfo {
    #[pyo3(get)]
    bits: u32,
    #[pyo3(get)]
    max: i128,
    #[pyo3(get)]
    min: i128,
    #[pyo3(get)]
    dtype: Py<Dtype>,
}

#[pymethods]
impl Iinfo {
    fn __repr__(&self) -> String {
        format!("iinfo(bits={}, max={}, min={})", self.bits, self.max, self.min)
    }
}

macro_rules! iinfo_for {
    ($t:ty, $bits:literal, $dtype:expr) => {
        Iinfo { bits: $bits, max: <$t>::MAX as i128, min: <$t>::MIN as i128, dtype: $dtype }
    };
}

pub fn iinfo_i8(dtype: Py<Dtype>) -> Iinfo {
    iinfo_for!(i8, 8, dtype)
}
pub fn iinfo_i16(dtype: Py<Dtype>) -> Iinfo {
    iinfo_for!(i16, 16, dtype)
}
pub fn iinfo_i32(dtype: Py<Dtype>) -> Iinfo {
    iinfo_for!(i32, 32, dtype)
}
pub fn iinfo_i64(dtype: Py<Dtype>) -> Iinfo {
    iinfo_for!(i64, 64, dtype)
}
pub fn iinfo_u8(dtype: Py<Dtype>) -> Iinfo {
    Iinfo { bits: 8, max: u8::MAX as i128, min: 0, dtype }
}
pub fn iinfo_u16(dtype: Py<Dtype>) -> Iinfo {
    Iinfo { bits: 16, max: u16::MAX as i128, min: 0, dtype }
}
pub fn iinfo_u32(dtype: Py<Dtype>) -> Iinfo {
    Iinfo { bits: 32, max: u32::MAX as i128, min: 0, dtype }
}
pub fn iinfo_u64(dtype: Py<Dtype>) -> Iinfo {
    Iinfo { bits: 64, max: u64::MAX as i128, min: 0, dtype }
}
