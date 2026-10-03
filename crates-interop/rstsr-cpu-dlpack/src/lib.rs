//! DLPack interchange for rstsr CPU tensors.
//!
//! The crate is pure Rust: it converts between rstsr tensors and DLPack managed
//! tensors and knows nothing about Python. `PyCapsule` boxing stays with the
//! host (a Python-embedding Rust program, or a small ctypes shim); the host
//! only has to hand raw pointers to this crate and call the deleter stored in
//! an export.
//!
//! # Directions
//!
//! - **export**: [`into_dlpack`] (move), [`to_dlpack_shared`] (zero-copy, read-only share),
//!   [`to_dlpack_shared_view`] (zero-copy share of a basic-indexed view), [`to_dlpack_copy`] (copy
//!   fallback for views) — each has a fallible `_f` variant.
//! - **import**: [`from_dlpack_versioned_f`], [`from_dlpack_legacy_f`] → read-only zero-copy
//!   [`TensorDlpack`] that owns the producer's lifetime; `kDLBool` payloads are validated to be 0
//!   or 1 (Rust `bool` has no other values).
//!
//! # Cargo features
//!
//! - `half` (default): `f16`/`bf16` dtype support (via the `half` crate).
//! - `row_major` / `col_major`, `std`: forwarded to `rstsr-core` and `rstsr-common`.
//!
//! # Prelude
//!
//! [`prelude`] groups the items above into the workspace's `rstsr_traits` /
//! `rstsr_structs` / `rstsr_funcs` namespaces. The `rstsr` facade forwards them under its
//! `dlpack` feature, so they are also reachable as `rt::dlpack::*`.
//!
//! # Interchange version
//!
//! Exports speak the DLPack 1.0 subset (`version = {1, 0}`) although the
//! `dlpack-ffi` header is 1.3; NumPy 2.5 consumes that subset and itself
//! exports `{1, 0}`. All CPU devices (`Raw = Vec<T>`) are mapped to `kDLCPU`.
//!
//! # Example
//!
//! ```
//! use rstsr_cpu_dlpack::*;
//! use rstsr_core::prelude::*;
//!
//! let device = DeviceCpuSerial::default();
//! let tensor: Tensor<f64, DeviceCpuSerial, IxD> = rt::arange_f((0.0, 5.0, 1.0, &device))?;
//! let shared = into_shared_dlpack_f(tensor)?;
//! let export = to_dlpack_shared(&shared);
//! assert_eq!(export.flags(), dlpack_ffi::DLPACK_FLAG_BITMASK_READ_ONLY as u64);
//! # Ok::<(), rstsr_common::error::Error>(())
//! ```

pub mod device;
pub mod dtype;
pub mod export;
pub mod import;
pub mod prelude;
pub mod repr;

pub use dlpack_ffi;
/// Re-export of [`rstsr_common::rstsr_raise`], used by this crate's exported macros.
pub use rstsr_common::rstsr_raise;

pub use crate::device::DeviceDlpackAPI;
pub use crate::dtype::DlpackDtype;
pub use crate::export::{
    into_dlpack, into_dlpack_f, to_dlpack_copy, to_dlpack_copy_f, to_dlpack_shared, to_dlpack_shared_f,
    to_dlpack_shared_view, to_dlpack_shared_view_f, DlpackExport, DlpackSharedBaseAPI,
};
pub use crate::import::{from_dlpack_legacy_f, from_dlpack_versioned_f};
pub use crate::repr::{into_shared_dlpack_f, DataDlpack, DlpackForeignOwner, TensorDlpack, TensorDlpackShared};
