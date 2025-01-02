pub mod rstsr_traits {
    pub use crate::layout::{
        DimAPI, DimBaseAPI, DimBroadcastableAPI, DimDevAPI, DimIntoAPI, DimLargerOneAPI,
        DimLayoutContigAPI, DimMaxAPI, DimShapeAPI, DimSmallerOneAPI, DimStrideAPI,
    };
    pub use crate::tensor::asarray::AsArrayAPI;
    pub use crate::tensor::creation::{
        ArangeAPI, EmptyAPI, EmptyLikeAPI, EyeAPI, FullAPI, FullLikeAPI, LinspaceAPI, OnesAPI,
        OnesLikeAPI, ZerosAPI, ZerosLikeAPI,
    };
    pub use crate::tensor::creation_from_tensor::DiagAPI;
    pub use crate::tensor::indexing::{TensorSliceAPI, TensorSliceMutAPI};
    pub use core::ops::*;
}

pub mod rstsr_structs {
    pub use crate::device_cpu_serial::device::DeviceCpuSerial;
    #[cfg(feature = "faer")]
    pub use crate::device_faer::device::DeviceFaer;
    pub use crate::DeviceCpu;

    pub use crate::error::{Error, Result};
    pub use crate::layout::indexer::{Ellipsis, NewAxis};
    pub use crate::layout::{Ix, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, Ix7, Ix8, Ix9, IxD, IxDyn, Layout};
    pub use crate::{
        Tensor, TensorArc, TensorBase, TensorCow, TensorMut, TensorRef, TensorView, TensorViewMut,
    };
}

pub mod rstsr_funcs {
    pub use crate::tensor::asarray::{asarray, asarray_f};
    pub use crate::tensor::creation::{
        arange, arange_f, empty, empty_f, empty_like, empty_like_f, eye, eye_f, full, full_f,
        full_like, full_like_f, linspace, linspace_f, ones, ones_f, ones_like, ones_like_f, zeros,
        zeros_f, zeros_like, zeros_like_f,
    };
    pub use crate::tensor::creation_from_tensor::{diag, diag_f};
    pub use crate::tensor::operators::{
        add, add_f, add_with_output, add_with_output_f, div, div_f, div_with_output,
        div_with_output_f, matmul, matmul_f, matmul_with_output, matmul_with_output_f, mul, mul_f,
        mul_with_output, mul_with_output_f, rem, rem_f, rem_with_output, rem_with_output_f, sub,
        sub_f, sub_with_output, sub_with_output_f,
    };
    pub use crate::tensor::operators::{
        add_assign, add_assign_f, div_assign, div_assign_f, mul_assign, mul_assign_f, rem_assign,
        rem_assign_f, sub_assign, sub_assign_f,
    };
    pub use crate::tensor::operators::{
        bitand, bitand_f, bitand_with_output, bitand_with_output_f, bitor, bitor_f,
        bitor_with_output, bitor_with_output_f, bitxor, bitxor_f, bitxor_with_output,
        bitxor_with_output_f, shl, shl_f, shl_with_output, shl_with_output_f, shr, shr_f,
        shr_with_output, shr_with_output_f,
    };
    pub use crate::tensor::operators::{
        bitand_assign, bitand_assign_f, bitor_assign, bitor_assign_f, bitxor_assign,
        bitxor_assign_f, shl_assign, shl_assign_f, shr_assign, shr_assign_f,
    };
    pub use crate::tensor::operators::{neg, neg_f, not, not_f};
}

pub mod rstsr_macros {
    pub use crate::{
        rstsr_assert, rstsr_assert_eq, rstsr_error, rstsr_invalid, rstsr_pattern, rstsr_raise,
    };
    pub use crate::{s, slice};
}

// final re-exports

pub use rstsr_funcs::*;
pub use rstsr_macros::*;
pub use rstsr_structs::*;
pub use rstsr_traits::*;

pub mod rstsr {
    pub use super::rstsr_funcs;
    pub use super::rstsr_macros;
    pub use super::rstsr_structs;
    pub use super::rstsr_traits;

    pub use super::rstsr_funcs::*;
    pub use super::rstsr_macros::*;
    pub use super::rstsr_structs::*;
    pub use super::rstsr_traits::*;
}

pub use rstsr as rt;
