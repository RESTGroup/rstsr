pub mod rstsr_traits {
    pub use crate::tensor::asarray::AsArrayAPI;
    pub use crate::tensor::creation::EmptyLikeAPI;
}

pub mod rstsr_structs {
    pub use crate::error::{Error, Result};
    pub use crate::layout::{Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, Ix7, Ix8, Ix9, IxD, IxDyn, Layout};
    pub use crate::{Tensor, TensorBase, TensorCow, TensorView, TensorViewMut};
}

pub mod rstsr_funcs {
    pub use crate::tensor::asarray::{asarray, asarray_f};
    pub use crate::tensor::creation::{empty_like, empty_like_f};
}

// final re-exports

pub use rstsr_traits::*;

pub mod rstsr {
    pub use super::rstsr_funcs::*;
    pub use super::rstsr_structs::*;
}
