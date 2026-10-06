//! Marker trait for valid element (scalar) dtypes.

use num::Complex;

/// Marker for valid rstsr element dtypes: booleans, integers, floats, complex
/// floats and half floats (the promotion-matrix dtype set).
///
/// Implemented explicitly per dtype - never via a blanket impl - so tensor
/// wrapper types (`&TensorAny`, `TensorView`, ...) cannot implement it.
/// Overloaded parameters that accept either a tensor or a scalar use this
/// bound to keep the overloads disjoint; unlike the `num::Num` bound used by
/// the arithmetic ops, `bool` is included.
pub trait DTypeScalarAPI: Sized + Send + Sync + Clone + 'static {}

macro_rules! impl_dtype_scalar {
    ($($T:ty),* $(,)?) => {
        $(impl DTypeScalarAPI for $T {})*
    };
}

impl_dtype_scalar!(bool, u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, f32, f64);
impl_dtype_scalar!(Complex<f32>, Complex<f64>);

#[cfg(feature = "half")]
impl_dtype_scalar!(half::f16, half::bf16);
