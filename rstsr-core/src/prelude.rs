pub mod rstsr_traits {
    pub use rstsr_common::prelude::rstsr_traits::*;

    pub use crate::storage::conversion::DeviceChangeAPI;
    pub use crate::storage::data::{DataAPI, DataCloneAPI, DataForceMutAPI, DataIntoCowAPI, DataMutAPI, DataOwnedAPI};
    pub use crate::storage::device::{DeviceAPI, DeviceBaseAPI, DeviceRawAPI, DeviceStorageAPI};
    pub use crate::tensor::asarray::AsArrayAPI;
    pub use crate::tensor::creation::{
        ArangeAPI, EmptyAPI, EmptyLikeAPI, EyeAPI, FullAPI, FullLikeAPI, LinspaceAPI, OnesAPI, OnesLikeAPI, TrilAPI,
        TriuAPI, ZerosAPI, ZerosLikeAPI,
    };
    pub use crate::tensor::creation_from_tensor::{
        ConcatAPI, DiagAPI, HStackAPI, MeshgridAPI, StackAPI, UnstackAPI, VStackAPI,
    };
    pub use crate::tensor::operators::op_binary_common::{
        TensorATan2API, TensorCopySignAPI, TensorEqualAPI, TensorFloorDivideAPI, TensorGreaterAPI,
        TensorGreaterEqualAPI, TensorHypotAPI, TensorLessAPI, TensorLessEqualAPI, TensorLogAddExpAPI, TensorMaximumAPI,
        TensorMinimumAPI, TensorNotEqualAPI, TensorPowAPI,
    };
    pub use crate::tensor::operators::op_unary_common::{
        TensorAbsAPI, TensorAcosAPI, TensorAcoshAPI, TensorAsinAPI, TensorAsinhAPI, TensorAtanAPI, TensorAtanhAPI,
        TensorCeilAPI, TensorConjAPI, TensorCosAPI, TensorCoshAPI, TensorExpAPI, TensorExpm1API, TensorFloorAPI,
        TensorImagAPI, TensorInvAPI, TensorIsFiniteAPI, TensorIsInfAPI, TensorIsNanAPI, TensorLog10API, TensorLog2API,
        TensorLogAPI, TensorRealAPI, TensorRoundAPI, TensorSignAPI, TensorSignBitAPI, TensorSinAPI, TensorSinhAPI,
        TensorSqrtAPI, TensorSquareAPI, TensorTanAPI, TensorTanhAPI, TensorTruncAPI,
    };
    pub use crate::tensor::ownership_conversion::{TensorIntoOwnedAPI, TensorViewAPI, TensorViewMutAPI};
    pub use crate::tensor::reduction::TensorSumBoolAPI;

    #[cfg(feature = "rayon")]
    pub use crate::feature_rayon::DeviceRayonAPI;
}

pub mod rstsr_structs {
    pub use rstsr_common::prelude::rstsr_structs::*;

    pub use crate::device_cpu_serial::device::DeviceCpuSerial;
    #[cfg(feature = "faer")]
    pub use crate::device_faer::device::DeviceFaer;
    pub use crate::DeviceCpu;

    pub use crate::tensor::tensor_mutable::TensorMutable;
    pub use crate::{
        Tensor, TensorAny, TensorArc, TensorBase, TensorCow, TensorMut, TensorRef, TensorReference, TensorView,
        TensorViewMut,
    };
}

pub mod rstsr_funcs {
    pub use crate::tensor::adv_indexing::{bool_select, bool_select_f, index_select, index_select_f};
    pub use crate::tensor::asarray::{asarray, asarray_f};
    pub use crate::tensor::creation::{
        arange, arange_f, assume_init, assume_init_f, empty, empty_f, empty_like, empty_like_f, eye, eye_f, full,
        full_f, full_like, full_like_f, linspace, linspace_f, ones, ones_f, ones_like, ones_like_f, tril, tril_f, triu,
        triu_f, uninit, uninit_f, zeros, zeros_f, zeros_like, zeros_like_f,
    };
    pub use crate::tensor::creation_from_tensor::{
        concat, concat_f, concatenate, concatenate_f, diag, diag_f, hstack, hstack_f, meshgrid, meshgrid_f, stack,
        stack_f, unstack, unstack_f, vstack, vstack_f,
    };
    pub use crate::tensor::indexing::{
        diagonal, diagonal_f, diagonal_mut, diagonal_mut_f, into_diagonal, into_diagonal_f, into_diagonal_mut,
        into_diagonal_mut_f, into_slice, into_slice_f, slice, slice_f, slice_mut, slice_mut_f,
    };
    pub use crate::tensor::manuplication::{
        broadcast_arrays, broadcast_arrays_f, change_contig, change_contig_f, change_layout, change_layout_f,
        change_shape, change_shape_f, expand_dims, expand_dims_f, flip, flip_f, into_broadcast, into_broadcast_f,
        into_contig, into_contig_f, into_dim, into_dim_f, into_dyn, into_expand_dims, into_expand_dims_f, into_flip,
        into_flip_f, into_layout, into_layout_f, into_permute_dims, into_permute_dims_f, into_reverse_axes, into_shape,
        into_shape_assume_contig, into_shape_assume_contig_f, into_shape_f, into_squeeze, into_squeeze_f,
        into_swapaxes, into_swapaxes_f, into_transpose, into_transpose_f, permute_dims, permute_dims_f, reshape,
        reshape_assume_contig, reshape_assume_contig_f, reshape_f, reverse_axes, squeeze, squeeze_f, swapaxes,
        swapaxes_f, to_broadcast, to_broadcast_f, to_contig, to_contig_f, to_dim, to_dim_f, to_dyn, to_layout,
        to_layout_f, to_shape, to_shape_assume_contig, to_shape_assume_contig_f, to_shape_f, transpose, transpose_f,
    };

    // binary arithmetics
    pub use crate::tensor::operators::{
        add, add_f, add_with_output, add_with_output_f, div, div_f, div_with_output, div_with_output_f, matmul,
        matmul_f, matmul_with_output, matmul_with_output_f, mul, mul_f, mul_with_output, mul_with_output_f, rem, rem_f,
        rem_with_output, rem_with_output_f, sub, sub_f, sub_with_output, sub_with_output_f,
    };
    // binary arithmetics with assignment
    pub use crate::tensor::operators::{
        add_assign, add_assign_f, div_assign, div_assign_f, mul_assign, mul_assign_f, rem_assign, rem_assign_f,
        sub_assign, sub_assign_f,
    };
    // binary bitwise
    pub use crate::tensor::operators::{
        bitand, bitand_f, bitand_with_output, bitand_with_output_f, bitor, bitor_f, bitor_with_output,
        bitor_with_output_f, bitxor, bitxor_f, bitxor_with_output, bitxor_with_output_f, shl, shl_f, shl_with_output,
        shl_with_output_f, shr, shr_f, shr_with_output, shr_with_output_f,
    };
    // binary bitwise with assignment
    pub use crate::tensor::operators::{
        bitand_assign, bitand_assign_f, bitor_assign, bitor_assign_f, bitxor_assign, bitxor_assign_f, shl_assign,
        shl_assign_f, shr_assign, shr_assign_f,
    };
    // unary arithmetics
    pub use crate::tensor::operators::{neg, neg_f, not, not_f};
    // unary common functions
    pub use crate::tensor::operators::{
        abs, abs_f, acos, acos_f, acosh, acosh_f, asin, asin_f, asinh, asinh_f, atan, atan_f, atanh, atanh_f, ceil,
        ceil_f, conj, conj_f, cos, cos_f, cosh, cosh_f, exp, exp_f, expm1, expm1_f, floor, floor_f, imag, imag_f, inv,
        inv_f, is_finite, is_finite_f, is_inf, is_inf_f, is_nan, is_nan_f, log, log10, log10_f, log2, log2_f, log_f,
        real, real_f, round, round_f, sign, sign_f, signbit, signbit_f, sin, sin_f, sinh, sinh_f, sqrt, sqrt_f, square,
        square_f, tan, tan_f, tanh, tanh_f, trunc, trunc_f,
    };
    // binary common functions
    pub use crate::tensor::operators::{
        atan2, atan2_f, copysign, copysign_f, eq, eq_f, equal, equal_f, equal_than, equal_than_f, floor_divide,
        floor_divide_f, ge, ge_f, greater, greater_equal, greater_equal_f, greater_equal_to, greater_equal_to_f,
        greater_f, greater_than, greater_than_f, gt, gt_f, hypot, hypot_f, le, le_f, less, less_equal, less_equal_f,
        less_equal_to, less_equal_to_f, less_f, less_than, less_than_f, log_add_exp, log_add_exp_f, lt, lt_f, maximum,
        maximum_f, minimum, minimum_f, ne, ne_f, not_equal, not_equal_f, not_equal_to, not_equal_to_f, pow, pow_f,
    };
    // reduction
    pub use crate::tensor::reduction::{
        argmax, argmax_all, argmax_all_f, argmax_axes, argmax_axes_f, argmax_f, argmin, argmin_all, argmin_all_f,
        argmin_axes, argmin_axes_f, argmin_f, l2_norm, l2_norm_all, l2_norm_all_f, l2_norm_axes, l2_norm_axes_f,
        l2_norm_f, max, max_all, max_all_f, max_axes, max_axes_f, max_f, mean, mean_all, mean_all_f, mean_axes,
        mean_axes_f, mean_f, min, min_all, min_all_f, min_axes, min_axes_f, min_f, prod, prod_all, prod_all_f,
        prod_axes, prod_axes_f, prod_f, std, std_all, std_all_f, std_axes, std_axes_f, std_f, sum, sum_all, sum_all_f,
        sum_axes, sum_axes_f, sum_f, unraveled_argmax, unraveled_argmax_all, unraveled_argmax_all_f,
        unraveled_argmax_axes, unraveled_argmax_axes_f, unraveled_argmax_f, unraveled_argmin, unraveled_argmin_all,
        unraveled_argmin_all_f, unraveled_argmin_axes, unraveled_argmin_axes_f, unraveled_argmin_f, var, var_all,
        var_all_f, var_axes, var_axes_f, var_f,
    };
}

pub mod rstsr_macros {
    pub use rstsr_common::prelude::rstsr_macros::*;
}

// final re-exports

pub use rstsr_macros::*;
pub use rstsr_structs::*;
pub use rstsr_traits::*;

#[allow(unused_imports)]
pub mod rt {
    pub use rstsr_common::prelude::rt::*;

    pub use super::rstsr_funcs;
    pub use super::rstsr_macros;
    pub use super::rstsr_structs;
    pub use super::rstsr_traits;

    pub use super::rstsr_funcs::*;
    pub use super::rstsr_macros::*;
    pub use super::rstsr_structs::*;
    pub use super::rstsr_traits::*;
}
