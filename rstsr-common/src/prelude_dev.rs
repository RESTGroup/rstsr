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

pub use crate::error::{Error, Result};
pub use crate::flags::*;
pub use crate::layout::*;
pub use crate::util::*;

#[cfg(feature = "rayon")]
pub use crate::par_iter::*;
#[cfg(feature = "rayon")]
pub use rayon::ThreadPool;

pub use crate::{
    impl_from_tuple_to_axes_index, rstsr_assert, rstsr_assert_eq, rstsr_errcode, rstsr_error,
    rstsr_invalid, rstsr_pattern, rstsr_raise, s, slice,
};
