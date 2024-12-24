pub mod rstsr_traits {
    pub use crate::tensor::asarray::AsArrayAPI;
    pub use crate::tensor::creation::{
        ArangeAPI, EmptyAPI, EmptyLikeAPI, EyeAPI, FullAPI, FullLikeAPI, LinspaceAPI, OnesAPI,
        OnesLikeAPI, ZerosAPI, ZerosLikeAPI,
    };
}

pub mod rstsr_structs {
    pub use crate::device_cpu_serial::device::DeviceCpuSerial;
    #[cfg(feature = "faer")]
    pub use crate::device_faer::device::DeviceFaer;
    pub use crate::DeviceCpu;

    pub use crate::error::{Error, Result};
    pub use crate::layout::{Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, Ix7, Ix8, Ix9, IxD, IxDyn, Layout};
    pub use crate::{Tensor, TensorBase, TensorCow, TensorView, TensorViewMut};
}

pub mod rstsr_funcs {
    pub use crate::tensor::asarray::{asarray, asarray_f};
    pub use crate::tensor::creation::{
        arange, arange_f, empty, empty_f, empty_like, empty_like_f, eye, eye_f, full, full_f,
        full_like, full_like_f, linspace, linspace_f, ones, ones_f, ones_like, ones_like_f, zeros,
        zeros_f, zeros_like, zeros_like_f,
    };
}

// final re-exports

pub use rstsr_traits::*;

pub mod rstsr {
    pub use super::rstsr_funcs::*;
    pub use super::rstsr_structs::*;
}
