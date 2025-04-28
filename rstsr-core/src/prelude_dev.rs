extern crate alloc;
pub use alloc::boxed::Box;
pub use alloc::format;
pub use alloc::string::{String, ToString};
pub use alloc::vec;
pub use alloc::vec::Vec;

pub use core::fmt::{Debug, Display, Write};
pub use core::marker::PhantomData;

pub use duplicate::{duplicate_item, substitute_item};
pub use itertools::{izip, Itertools};

#[cfg(feature = "rayon")]
pub use rayon::ThreadPool;

pub use rstsr_common::prelude_dev::*;

pub use rstsr_native_impl::prelude_dev::*;

pub use crate::storage::assignment::*;
pub use crate::storage::combined_trait::*;
pub use crate::storage::conversion::*;
pub use crate::storage::creation::*;
pub use crate::storage::data::*;
pub use crate::storage::device::*;
pub use crate::storage::matmul::*;
pub use crate::storage::operators::*;
pub use crate::storage::reduction::*;

pub use crate::device_cpu_serial::device::*;
pub use crate::DeviceCpu;

#[allow(unused_imports)]
pub use crate::dev_utilities::*;

pub use crate::tensor::asarray::*;
pub use crate::tensor::creation::*;
pub use crate::tensor::creation_from_tensor::*;
pub use crate::tensor::device_conversion::*;
pub use crate::tensor::ext_conversion::*;
pub use crate::tensor::iterator_axes::*;
pub use crate::tensor::iterator_elem::*;
pub use crate::tensor::manuplication::*;
pub use crate::tensor::ownership_conversion::*;
pub use crate::tensor::tensor_mutable::*;

#[cfg(feature = "rayon")]
pub use crate::feature_rayon::assignment::*;
#[cfg(feature = "rayon")]
pub use crate::feature_rayon::device::*;
#[cfg(feature = "rayon")]
pub use crate::feature_rayon::op_with_func::*;
#[cfg(feature = "rayon")]
pub use crate::feature_rayon::par_iter::*;

#[cfg(feature = "faer")]
pub use crate::device_faer::device::*;

pub use crate::tensorbase::*;
