extern crate alloc;
pub use alloc::boxed::Box;
pub use alloc::format;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec;

pub use core::fmt::{Debug, Display, Write};
pub use core::marker::PhantomData;
pub use core::mem::MaybeUninit;

pub use duplicate::{duplicate, duplicate_item, substitute_item};
pub use itertools::{izip, Itertools};

#[cfg(feature = "rayon")]
pub use rayon::ThreadPool;

pub use rstsr_common::prelude_dev::*;
pub use rstsr_dtype_traits::{DTypeCastAPI, DTypePromoteAPI, ExtFloat, ExtNum, ExtReal, IsCloseArgs};

pub use rstsr_native_impl::prelude_dev::*;

pub use crate::storage::exports::*;
pub use crate::tensor::exports::*;

pub use crate::device_cpu_serial::device::*;
pub use crate::DeviceCpu;

#[allow(unused_imports)]
pub use crate::dev_utilities::*;

pub use crate::prelude::rstsr_traits::*;

#[cfg(feature = "rayon")]
pub use crate::feature_rayon::device::*;

#[cfg(feature = "faer")]
pub use crate::device_faer::device::*;

pub use crate::tensorbase::*;

pub use crate::tensor_from_nested;
