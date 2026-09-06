//! This module is to declare API for external tensor objects to rstsr object.

/// API trait converting external tensor-like objects into rstsr tensors.
pub trait IntoRSTSR {
    type RSTSR;
    fn into_rstsr(self) -> Self::RSTSR;
}
