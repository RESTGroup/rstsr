#![cfg_attr(not(test), no_std)]
#![doc = include_str!("docs/lib.md")]
#![allow(unknown_lints)]
// This option is for myself as python-like developer.
#![allow(clippy::needless_return)]
// Resolution to something like `tensor.slice_mut() += scalar`.
// This code is not allowed in rust, but `*&mut tensor.slice_mut() += scalar` is allowed.
#![allow(clippy::deref_addrof)]
// Allow `a % b == 0` instead of `is_multiple_of`
// which is only stabilized after rustc 1.89
#![allow(clippy::manual_is_multiple_of)]

// this line is for docs
#[allow(unused_imports)]
use crate::prelude::rstsr_funcs::*;
#[allow(unused_imports)]
use crate::prelude::rstsr_structs::*;
#[allow(unused_imports)]
use crate::prelude::rstsr_traits::*;

pub mod prelude;
pub mod prelude_dev;

pub mod storage;
pub mod tensor;
pub mod tensorbase;
pub use tensorbase::{
    Tensor, TensorAny, TensorArc, TensorBase, TensorCow, TensorMut, TensorRef, TensorReference, TensorView,
    TensorViewMut,
};

pub mod format;

#[cfg(feature = "rayon")]
pub mod feature_rayon;

#[cfg(feature = "faer")]
pub mod device_faer;

pub mod device_cpu_serial;

mod dev_utilities;

#[cfg(feature = "faer_as_default")]
pub type DeviceCpu = device_faer::device::DeviceFaer;

#[cfg(not(feature = "faer_as_default"))]
pub type DeviceCpu = device_cpu_serial::DeviceCpuSerial;

pub mod array_api_standard {
    #![doc = include_str!("docs/array_api_standard.md")]
    #![allow(unused_imports)]

    use crate::prelude::*;
    use crate::prelude_dev::Indexer;
    use core::ops::*;
    use num::complex::ComplexFloat;
    use num::{pow::Pow, Float, Num, Signed};
    use rstsr_dtype_traits::*;
    use rt::*;
}

pub mod api_specification {
    #![doc = include_str!("docs/api_specification.md")]
    #![allow(unused_imports)]

    use crate::prelude::*;
    use rt::*;
    extern crate alloc;
}
