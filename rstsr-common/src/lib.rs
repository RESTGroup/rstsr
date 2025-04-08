#![allow(clippy::needless_return)]
// Resolution to something like `tensor.slice_mut() += scalar`.
// This code is not allowed in rust, but `*&mut tensor.slice_mut() += scalar` is allowed.
#![allow(clippy::deref_addrof)]

pub mod prelude;
pub mod prelude_dev;

pub mod error;
pub mod flags;
pub mod layout;
pub mod util;
pub mod format_layout;

#[cfg(feature = "rayon")]
pub mod par_iter;
