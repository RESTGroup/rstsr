//! Matrix rank on the BLAS devices (the array-API `linalg.matrix_rank`).
//!
//! A direct port of the faer entry: the rank of a matrix is the number of its
//! singular values strictly greater than `rtol * s_max`, where `s_max` is its
//! largest singular value. [`MatrixRankRtol`] carries the tolerance: `None`
//! uses the default `max(M, N) * eps` (`eps` the input's real machine
//! epsilon), `Float` a scalar for every matrix, and `Array` one tolerance per
//! matrix broadcast against the stack — the array's device is checked against
//! the input's, the scalar needs no device. Singular values come from
//! [`super::svdvals::blas_impl_svdvals_f`] (GESDD without `U`/`Vᵀ`). Unlike
//! NumPy there is no Hermitian fast path: every matrix is ranked from its SVD.

use crate::DeviceBLAS;
use num::Float;
use num::FromPrimitive;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

use super::svdvals::blas_impl_svdvals_f;

/// n-dim `matrix_rank` on the BLAS devices: the rank of every matrix slice of
/// the stack, `usize`-typed and batch-shaped.
///
/// The matrix axes are the last two shape axes under a row-major device and
/// the first two under a column-major one; the rest is the batch, and an
/// array tolerance is broadcast against it in that same device order. The
/// tolerance is validated up front (device match, and a shape that broadcasts
/// to the batch shape) before the empty-input short-circuit and the SVD, so a
/// bad tolerance errors on an empty input too.
pub fn blas_impl_matrix_rank_f<T>(
    a: TensorView<'_, T, DeviceBLAS, IxD>,
    rtol: MatrixRankRtol<T::Real, DeviceBLAS>,
) -> Result<Tensor<usize, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    T::Real: Float + FromPrimitive,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationAnyAPI<usize>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpMaxAPI<T::Real, IxD, TOut = T::Real>
        + OpMulAPI<T::Real, T::Real, T::Real, IxD>
        + OpGreaterAPI<T::Real, T::Real, IxD, TOut = bool>
        + OpSumBoolAPI<IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, inner) = stack_split_shape(a.shape(), 2, order)?;
    let [m, n] = [inner[0], inner[1]];

    // validate `rtol` up front, before the empty-input short-circuit below (and
    // the SVD) can skip it: a bad tolerance errors on an empty input too
    if let MatrixRankRtol::Array(t) = &rtol {
        rstsr_assert!(t.device().same_device(&device), DeviceMismatch, "matrix_rank: rtol is on a different device")?;
        let shape = broadcast_shapes_f(&[t.shape().to_vec(), batch_shape.clone()], order)?;
        rstsr_assert_eq!(
            shape.as_slice(),
            batch_shape.as_slice(),
            InvalidLayout,
            "matrix_rank: rtol shape {:?} must broadcast to the batch shape {:?}",
            t.shape(),
            batch_shape
        )?;
    }

    // zero-size matrix axes give rank 0; an empty batch stays empty
    if m == 0 || n == 0 || batch_shape.contains(&0) {
        return zeros_f((batch_shape, &device));
    }

    // the singular-value axis: trailing under `RowMajor`, leading under `ColMajor`
    let k_ax: isize = match order {
        RowMajor => -1,
        ColMajor => 0,
    };
    let default_scale = T::Real::epsilon() * T::Real::from_usize(Ord::max(m, n)).unwrap();

    let s = blas_impl_svdvals_f(a.into())?;

    let smax = max_with_args_f(&s, (k_ax, true))?;
    // `tol` carries one tolerance per matrix, shaped like `smax` (the batch
    // axes plus a singleton on the singular-value axis)
    let tol: Tensor<T::Real, DeviceBLAS, IxD> = match rtol {
        MatrixRankRtol::None => &smax * default_scale,
        MatrixRankRtol::Float(f) => &smax * f,
        MatrixRankRtol::Array(t) => {
            // broadcast the batch tolerance (validated above), then re-insert
            // the singular-value singleton
            let target = stack_shape(&batch_shape, &[1], order);
            let rb = into_broadcast_f(t.view(), batch_shape)?;
            let r_exp = rb.into_shape_f(target)?;
            &smax * &r_exp
        },
    };
    // count the singular values above the per-matrix threshold
    let mask = greater_f(&s, &tol)?;
    mask.sum_with_args_f((k_ax, false))
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType> MatrixRankAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    T::Real: Float + FromPrimitive,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationAnyAPI<usize>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpMaxAPI<T::Real, IxD, TOut = T::Real>
        + OpMulAPI<T::Real, T::Real, T::Real, IxD>
        + OpGreaterAPI<T::Real, T::Real, IxD, TOut = bool>
        + OpSumBoolAPI<IxD>,
{
    type Out = Tensor<usize, DeviceBLAS, IxD>;
    fn matrix_rank_f(self) -> Result<Self::Out> {
        blas_impl_matrix_rank_f(self.view().to_dyn(), MatrixRankRtol::None)
    }
}

#[duplicate_item(
    ImplType                          Tr                                  ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [Tensor<T, DeviceBLAS, D>        ];
)]
impl<ImplType, A> MatrixRankAPI<DeviceBLAS> for (Tr, A)
where
    T: BlasFloat,
    T::Real: Float + FromPrimitive,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceCreationAnyAPI<T::Real>
        + DeviceCreationAnyAPI<usize>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceAPI<bool, Raw = Vec<bool>>
        + OpAssignArbitaryAPI<T::Real, Ix1, Ix1>
        + OpAssignAPI<T::Real, Ix1>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>
        + OpMaxAPI<T::Real, IxD, TOut = T::Real>
        + OpMulAPI<T::Real, T::Real, T::Real, IxD>
        + OpGreaterAPI<T::Real, T::Real, IxD, TOut = bool>
        + OpSumBoolAPI<IxD>,
    A: TryInto<MatrixRankRtol<T::Real, DeviceBLAS>>,
    A::Error: Into<Error>,
{
    type Out = Tensor<usize, DeviceBLAS, IxD>;
    fn matrix_rank_f(self) -> Result<Self::Out> {
        let (a, rtol) = self;
        blas_impl_matrix_rank_f(a.view().to_dyn(), rtol.try_into().map_err(Into::into)?)
    }
}
