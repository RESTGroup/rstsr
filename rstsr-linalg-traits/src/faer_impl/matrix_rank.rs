//! Matrix rank on the faer device (the array-API `linalg.matrix_rank`).
//!
//! The rank of a matrix is the number of its singular values strictly greater
//! than `rtol * s_max`, where `s_max` is its largest singular value.
//! [`MatrixRankRtol`] carries the tolerance: `None` uses the default
//! `max(M, N) * eps` (`eps` the input's real machine epsilon), `Float` a scalar
//! for every matrix, and `Array` one tolerance per matrix broadcast against the
//! stack — the array's device is checked against the input's, the scalar needs
//! no device. Singular values come from [`faer_impl_svdvals_f`]. Unlike NumPy
//! there is no Hermitian fast path: every matrix is ranked from its SVD.
//!
//! The two matrix axes are the input's **trailing** axes under a row-major device
//! and its **leading** axes under a column-major one; the batch is the rest, and
//! an `Array` tolerance is broadcast against it in that same device order (so
//! under row-major the alignment matches NumPy's trailing rule, under
//! column-major it mirrors it).
//!
//! A matrix axis of size zero yields rank `0` for every matrix (the stack
//! collapses to a single zero rank per matrix), matching NumPy.

use crate::faer_impl::batch::{batch_and_matrix_shape, stack_shape, with_parallel};
use crate::faer_impl::norm::NormReal;
use crate::faer_impl::svdvals::faer_impl_svdvals_f;
use crate::traits_def::{MatrixRankAPI, MatrixRankRtol};
use faer::traits::ComplexField;
use num::complex::ComplexFloat;
use num::Float;
use num::FromPrimitive;
use rstsr_core::prelude_dev::*;

pub fn faer_impl_matrix_rank_f<T, TR>(
    a: TensorView<'_, T, DeviceFaer, IxD>,
    rtol: MatrixRankRtol<TR, DeviceFaer>,
) -> Result<Tensor<usize, DeviceFaer, IxD>>
where
    T: ComplexField<Real = TR>
        + ExtNum<AbsOut = TR>
        + ComplexFloat<Real = TR>
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + 'static,
    TR: NormReal + Float + DTypePromoteAPI<TR, Res = TR>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let (batch_shape, [m, n]) = batch_and_matrix_shape(a.shape(), order)?;

    // validate `rtol` up front, before the empty-input short-circuit below (and
    // the SVD) can skip it: a bad tolerance errors on an empty input too
    if let MatrixRankRtol::Array(t) = &rtol {
        rstsr_assert!(t.device().same_device(&device), DeviceMismatch, "matrix_rank: rtol is on a different device")?;
        broadcast_shapes_f(&[t.shape().to_vec(), batch_shape.clone()], order)?;
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
    let default_scale = TR::epsilon() * TR::from_usize(Ord::max(m, n)).unwrap();

    let s = faer_impl_svdvals_f(a)?;

    with_parallel(&device, || {
        let smax = max_with_args_f(&s, (k_ax, true))?;
        // `tol` carries one tolerance per matrix, shaped like `smax` (the batch
        // axes plus a singleton on the singular-value axis)
        let tol: Tensor<TR, DeviceFaer, IxD> = match rtol {
            MatrixRankRtol::None => smax.mapv(|x| x * default_scale),
            MatrixRankRtol::Float(f) => smax.mapv(|x| x * f),
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
    })
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR> MatrixRankAPI<DeviceFaer> for Tr
where
    T: ComplexField<Real = TR>
        + ExtNum<AbsOut = TR>
        + ComplexFloat<Real = TR>
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + 'static,
    TR: NormReal + Float + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
{
    type Out = Tensor<usize, DeviceFaer, IxD>;
    fn matrix_rank_f(self) -> Result<Self::Out> {
        faer_impl_matrix_rank_f(self.view().to_dyn(), MatrixRankRtol::None)
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceFaer, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceFaer, D>];
   [T, D                           ] [Tensor<T, DeviceFaer, D>        ];
)]
impl<ImplType, TR, A> MatrixRankAPI<DeviceFaer> for (Tr, A)
where
    T: ComplexField<Real = TR>
        + ExtNum<AbsOut = TR>
        + ComplexFloat<Real = TR>
        + FromPrimitive
        + Clone
        + Send
        + Sync
        + 'static,
    TR: NormReal + Float + DTypePromoteAPI<TR, Res = TR>,
    D: DimAPI,
    A: TryInto<MatrixRankRtol<TR, DeviceFaer>>,
    A::Error: Into<Error>,
{
    type Out = Tensor<usize, DeviceFaer, IxD>;
    fn matrix_rank_f(self) -> Result<Self::Out> {
        let (a, rtol) = self;
        faer_impl_matrix_rank_f(a.view().to_dyn(), rtol.try_into().map_err(Into::into)?)
    }
}
