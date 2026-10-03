//! DLPack dtype mapping table and the dynamic dispatch macro.

use dlpack_ffi::DLDataType;
#[cfg(feature = "half")]
use rstsr_cpu_dlpack::dtype::{bf16, f16};
use rstsr_cpu_dlpack::dtype::{Complex32, Complex64};
use rstsr_cpu_dlpack::{with_dlpack_dtype, DlpackDtype};

fn dltype<T: DlpackDtype>() -> (u8, u8, u16) {
    let d = T::DLTYPE;
    (d.code, d.bits, d.lanes)
}

#[test]
fn dtype_table() {
    // code: int 0, uint 1, float 2, bfloat 4, complex 5, bool 6
    assert_eq!(dltype::<i8>(), (0, 8, 1));
    assert_eq!(dltype::<i16>(), (0, 16, 1));
    assert_eq!(dltype::<i32>(), (0, 32, 1));
    assert_eq!(dltype::<i64>(), (0, 64, 1));
    assert_eq!(dltype::<i128>(), (0, 128, 1));
    assert_eq!(dltype::<u8>(), (1, 8, 1));
    assert_eq!(dltype::<u16>(), (1, 16, 1));
    assert_eq!(dltype::<u32>(), (1, 32, 1));
    assert_eq!(dltype::<u64>(), (1, 64, 1));
    assert_eq!(dltype::<u128>(), (1, 128, 1));
    #[cfg(feature = "half")]
    assert_eq!(dltype::<f16>(), (2, 16, 1));
    assert_eq!(dltype::<f32>(), (2, 32, 1));
    assert_eq!(dltype::<f64>(), (2, 64, 1));
    #[cfg(feature = "half")]
    assert_eq!(dltype::<bf16>(), (4, 16, 1));
    assert_eq!(dltype::<Complex32>(), (5, 64, 1));
    assert_eq!(dltype::<Complex64>(), (5, 128, 1));
    assert_eq!(dltype::<bool>(), (6, 8, 1));
}

#[test]
fn from_dltype_checks_exact_match() {
    assert!(f32::from_dltype(f32::DLTYPE).is_ok());
    let wrong_bits = DLDataType { code: 2, bits: 64, lanes: 1 };
    assert!(f32::from_dltype(wrong_bits).is_err());
    let vector_lanes = DLDataType { code: 2, bits: 32, lanes: 4 };
    assert!(f32::from_dltype(vector_lanes).is_err());
}

#[test]
fn dispatch_macro_maps_codes() {
    let f32_dt = DLDataType { code: 2, bits: 32, lanes: 1 };
    let bits: usize = with_dlpack_dtype!(f32_dt, |T| core::mem::size_of::<T>()).unwrap();
    assert_eq!(bits, 4);

    let c64_dt = DLDataType { code: 5, bits: 128, lanes: 1 };
    let bits: usize = with_dlpack_dtype!(c64_dt, |T| core::mem::size_of::<T>()).unwrap();
    assert_eq!(bits, 16);

    let subbyte = DLDataType { code: 10, bits: 8, lanes: 1 };
    let res: rstsr_common::error::Result<usize> = with_dlpack_dtype!(subbyte, |T| core::mem::size_of::<T>());
    assert!(res.is_err());
}

#[cfg(feature = "half")]
#[test]
fn dispatch_macro_half_dtypes() {
    let f16_dt = DLDataType { code: 2, bits: 16, lanes: 1 };
    let bits: usize = with_dlpack_dtype!(f16_dt, |T| core::mem::size_of::<T>()).unwrap();
    assert_eq!(bits, 2);

    let bf16_dt = DLDataType { code: 4, bits: 16, lanes: 1 };
    let bits: usize = with_dlpack_dtype!(bf16_dt, |T| core::mem::size_of::<T>()).unwrap();
    assert_eq!(bits, 2);
}

#[cfg(not(feature = "half"))]
#[test]
fn dispatch_macro_half_dtypes_require_feature() {
    let f16_dt = DLDataType { code: 2, bits: 16, lanes: 1 };
    let bf16_dt = DLDataType { code: 4, bits: 16, lanes: 1 };
    let res: rstsr_common::error::Result<usize> = with_dlpack_dtype!(f16_dt, |T| core::mem::size_of::<T>());
    assert!(res.is_err());
    let res: rstsr_common::error::Result<usize> = with_dlpack_dtype!(bf16_dt, |T| core::mem::size_of::<T>());
    assert!(res.is_err());
}
