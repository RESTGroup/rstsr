//! This module is to declare API for external tensor objects to rstsr object.

/// API trait converting external tensor-like objects into rstsr tensors.
///
/// The ownership semantics of the conversion (zero-copy view vs. copying into
/// an owning tensor) depend on the source type; they are documented on each
/// specific implementation.
pub trait IntoRSTSR {
    type RSTSR;
    fn into_rstsr(self) -> Self::RSTSR;
}
