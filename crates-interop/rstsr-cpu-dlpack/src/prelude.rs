//! User-facing prelude of rstsr-cpu-dlpack.
//!
//! - [`rstsr_traits`]: the API traits ([`DlpackDtype`](crate::dtype::DlpackDtype),
//!   [`DlpackSharedBaseAPI`](crate::export::DlpackSharedBaseAPI), ...).
//! - [`rstsr_structs`]: the tensor types and the export handle
//!   ([`TensorDlpack`](crate::repr::TensorDlpack), [`DlpackExport`](crate::export::DlpackExport),
//!   ...).
//! - [`rstsr_funcs`]: free functions (the `rt::dlpack::` surface: `into_dlpack`,
//!   `from_dlpack_versioned_f`, ...).
//!
//! The `rstsr` facade crate forwards these namespaces, so with the `dlpack` feature of `rstsr`
//! the same items are reachable as `rt::dlpack::*`.

pub mod rstsr_traits {
    pub use crate::device::DeviceDlpackAPI;
    pub use crate::dtype::DlpackDtype;
    pub use crate::export::DlpackSharedBaseAPI;
}

pub mod rstsr_structs {
    pub use crate::export::DlpackExport;
    pub use crate::repr::{DataDlpack, DlpackForeignOwner, TensorDlpack, TensorDlpackShared};
}

pub mod rstsr_funcs {
    pub use crate::export::{
        into_dlpack, into_dlpack_f, to_dlpack_copy, to_dlpack_copy_f, to_dlpack_shared, to_dlpack_shared_f,
        to_dlpack_shared_view, to_dlpack_shared_view_f,
    };
    pub use crate::import::{from_dlpack_legacy_f, from_dlpack_versioned_f};
    pub use crate::repr::into_shared_dlpack_f;
}
