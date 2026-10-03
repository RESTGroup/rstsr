//! DLPack data types and the Rust element types they map to.

use dlpack_ffi::{DLDataType, DLDataTypeCode};
use rstsr_common::error::Result;
use rstsr_common::prelude::rstsr_macros::rstsr_raise;

#[cfg(feature = "half")]
pub use half::{bf16, f16};
pub use num::Complex;

/// Complex with two `f32` components (DLPack `kDLComplex`, 64 bits).
pub type Complex32 = Complex<f32>;
/// Complex with two `f64` components (DLPack `kDLComplex`, 128 bits).
pub type Complex64 = Complex<f64>;

/// Rust element type expressible as a DLPack [`DLDataType`] with a single lane.
///
/// Implemented for the numeric dtypes rstsr supports. `f16`/`bf16` require the
/// `half` feature (default); `i128`/`u128` are valid DLPack but stock NumPy
/// cannot consume them (see the crate docs).
pub trait DlpackDtype: Copy + 'static {
    /// The DLPack data type of `Self`.
    const DLTYPE: DLDataType;

    /// Check that `dtype` is exactly `Self::DLTYPE`.
    fn from_dltype(dtype: DLDataType) -> Result<()> {
        let expected = Self::DLTYPE;
        let same = dtype.code == expected.code && dtype.bits == expected.bits && dtype.lanes == expected.lanes;
        if same {
            return Ok(());
        }
        rstsr_raise!(
            InvalidValue,
            "DLPack dtype {{code: {}, bits: {}, lanes: {}}} does not match the requested Rust type {{code: {}, bits: {}, lanes: {}}}",
            dtype.code,
            dtype.bits,
            dtype.lanes,
            expected.code,
            expected.bits,
            expected.lanes
        )
    }
}

macro_rules! impl_dlpack_dtype {
    ($code:expr => $($T:ty),* $(,)?) => {
        $(
            impl DlpackDtype for $T {
                const DLTYPE: DLDataType = DLDataType {
                    code: $code,
                    bits: (core::mem::size_of::<$T>() * 8) as u8,
                    lanes: 1,
                };
            }
        )*
    };
}

/// DLPack dtype code `kDLInt` (signed integers).
pub const CODE_INT: u8 = DLDataTypeCode::kDLInt.0 as u8;
/// DLPack dtype code `kDLUInt` (unsigned integers).
pub const CODE_UINT: u8 = DLDataTypeCode::kDLUInt.0 as u8;
/// DLPack dtype code `kDLFloat` (binary floating point).
pub const CODE_FLOAT: u8 = DLDataTypeCode::kDLFloat.0 as u8;
/// DLPack dtype code `kDLBfloat` (bfloat16).
pub const CODE_BFLOAT: u8 = DLDataTypeCode::kDLBfloat.0 as u8;
/// DLPack dtype code `kDLComplex` (complex floating point).
pub const CODE_COMPLEX: u8 = DLDataTypeCode::kDLComplex.0 as u8;
/// DLPack dtype code `kDLBool` (booleans, 8 bits in DLPack).
pub const CODE_BOOL: u8 = DLDataTypeCode::kDLBool.0 as u8;

impl_dlpack_dtype!(CODE_INT => i8, i16, i32, i64, i128);
impl_dlpack_dtype!(CODE_UINT => u8, u16, u32, u64, u128);
impl_dlpack_dtype!(CODE_FLOAT => f32, f64);
impl_dlpack_dtype!(CODE_COMPLEX => Complex32, Complex64);
impl_dlpack_dtype!(CODE_BOOL => bool);
#[cfg(feature = "half")]
impl_dlpack_dtype!(CODE_FLOAT => f16);
#[cfg(feature = "half")]
impl_dlpack_dtype!(CODE_BFLOAT => bf16);

/// One arm of [`with_dlpack_dtype!`] for the half-precision dtypes.
///
/// Two definitions exist because `#[cfg]` inside an exported macro is evaluated
/// in the *calling* crate: the `half` choice has to be made here, at the
/// definition. Without the feature the arm is an error and never expands `$body`.
#[cfg(feature = "half")]
#[doc(hidden)]
#[macro_export]
macro_rules! __with_dlpack_dtype_half_arm {
    ($t:ident, $T:ident, $body:expr, $dtype:expr) => {{
        type $T = $crate::dtype::$t;
        ::core::result::Result::Ok($body)
    }};
}

/// The `half`-disabled definition of [`__with_dlpack_dtype_half_arm!`].
#[cfg(not(feature = "half"))]
#[doc(hidden)]
#[macro_export]
macro_rules! __with_dlpack_dtype_half_arm {
    ($t:ident, $T:ident, $body:expr, $dtype:expr) => {{
        $crate::rstsr_raise!(
            UnImplemented,
            "DLPack dtype {{code: {}, bits: {}, lanes: {}}} requires the `half` feature of rstsr-cpu-dlpack",
            $dtype.code,
            $dtype.bits,
            $dtype.lanes
        )
    }};
}

/// Dispatch a body over the Rust type matching a DLPack data type.
///
/// Evaluates to `Result<R>`: `Ok(body)` for every supported dtype, `Err` for a
/// dtype the crate cannot represent (vector lanes, sub-byte floats, opaque
/// handles; `f16`/`bf16` when the `half` feature is disabled). The body must be
/// valid for each `T`; the bound `T: DlpackDtype` holds by construction.
///
/// ```
/// use rstsr_cpu_dlpack::dtype::CODE_FLOAT;
/// use rstsr_cpu_dlpack::dlpack_ffi::DLDataType;
///
/// let dtype = DLDataType { code: CODE_FLOAT, bits: 64, lanes: 1 };
/// let size: usize = rstsr_cpu_dlpack::with_dlpack_dtype!(dtype, |T| core::mem::size_of::<T>())?;
/// assert_eq!(size, 8);
/// # Ok::<(), rstsr_common::error::Error>(())
/// ```
#[macro_export]
macro_rules! with_dlpack_dtype {
    ($dtype:expr, |$T:ident| $body:expr $(,)?) => {{
        let dtype: $crate::dlpack_ffi::DLDataType = $dtype;
        match (dtype.code, dtype.bits, dtype.lanes) {
            ($crate::dtype::CODE_INT, 8, 1) => {
                type $T = i8;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_INT, 16, 1) => {
                type $T = i16;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_INT, 32, 1) => {
                type $T = i32;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_INT, 64, 1) => {
                type $T = i64;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_INT, 128, 1) => {
                type $T = i128;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_UINT, 8, 1) => {
                type $T = u8;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_UINT, 16, 1) => {
                type $T = u16;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_UINT, 32, 1) => {
                type $T = u32;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_UINT, 64, 1) => {
                type $T = u64;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_UINT, 128, 1) => {
                type $T = u128;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_FLOAT, 16, 1) => $crate::__with_dlpack_dtype_half_arm!(f16, $T, $body, dtype),
            ($crate::dtype::CODE_FLOAT, 32, 1) => {
                type $T = f32;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_FLOAT, 64, 1) => {
                type $T = f64;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_BFLOAT, 16, 1) => $crate::__with_dlpack_dtype_half_arm!(bf16, $T, $body, dtype),
            ($crate::dtype::CODE_COMPLEX, 64, 1) => {
                type $T = $crate::dtype::Complex32;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_COMPLEX, 128, 1) => {
                type $T = $crate::dtype::Complex64;
                ::core::result::Result::Ok($body)
            },
            ($crate::dtype::CODE_BOOL, 8, 1) => {
                type $T = bool;
                ::core::result::Result::Ok($body)
            },
            _ => $crate::rstsr_raise!(
                UnImplemented,
                "unsupported DLPack dtype {{code: {}, bits: {}, lanes: {}}}",
                dtype.code,
                dtype.bits,
                dtype.lanes
            ),
        }
    }};
}
