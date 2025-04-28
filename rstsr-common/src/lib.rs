#![allow(clippy::needless_return)]
// Resolution to something like `tensor.slice_mut() += scalar`.
// This code is not allowed in rust, but `*&mut tensor.slice_mut() += scalar` is allowed.
#![allow(clippy::deref_addrof)]

pub mod prelude;
pub mod prelude_dev;

pub mod alloc_vec;
pub mod axis_index;
pub mod error;
pub mod flags;
pub mod format_layout;
pub mod layout;
pub mod util;

#[cfg(feature = "rayon")]
pub mod par_iter;

#[cfg(all(not(clippy), feature = "col_major", feature = "row_major"))]
compile_error!("Cargo features col_major and row_major are mutually exclusive. Please enable only one of them.");
