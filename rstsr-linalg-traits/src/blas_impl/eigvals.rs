//! n-dim `eigvals` on the BLAS device: the eigenvalues of every square matrix
//! slice of the stack (the two matrix axes are the last two shape axes under a
//! row-major device, the first two under a column-major one; the rest is the
//! batch), or of the generalized problem `A v = λ B v` when a second matrix
//! is given (the batch dims of `a` and `b` broadcast against each other).
//!
//! The spectrum is always complex (`Complex<T::Real>`), in LAPACK's order and
//! unsorted, shaped `(..., M)` — or `(..., 2, M)` holding `(alpha, beta)` with
//! the [`Homogeneous`] marker (`beta` identically `1` for the standard
//! problem). The drivers run with both eigenvector flags off (`GEEV`/`GGEV`
//! values only); the inputs are consumed as scratch and must be treated as
//! destroyed.
//!
//! Overloads (every matrix slot takes `&TensorAny`, `TensorView`, `Tensor` or
//! `TensorMut`): `eigvals(a)`, `eigvals((a, Homogeneous))`, `eigvals((a, b))`,
//! `eigvals((a, b, Homogeneous))`.
//!
//! See also [`eig`](crate::linalg_auto_impl::eig).

use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `eigvals`: [`ref_impl_eig_nd_f`] with
/// [`EigSide::None`] (no eigenvector block is computed or allocated).
#[allow(clippy::type_complexity)]
pub fn blas_impl_eigvals_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    b: Option<TensorReference<'_, T, DeviceBLAS, IxD>>,
    homogeneous: bool,
) -> Result<Tensor<Complex<T::Real>, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    let (eigenvalues, left, right) = ref_impl_eig_nd_f(a, b, EigSide::None, homogeneous)?;
    debug_assert!(left.is_none() && right.is_none());
    Ok(eigenvalues)
}

/* #region full-args */

#[duplicate_item(
    ImplType  ToRef  Tr  ;
    [T, D, Ra: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [a.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [a.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [a.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalsAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    type Out = Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>;
    #[allow(unused_mut)]
    fn eigvals_f(self) -> Result<Self::Out> {
        let mut a = self;
        Ok(blas_impl_eigvals_f(ToRef, None, false)?.into_dim::<D::SmallerOne>())
    }
}

#[duplicate_item(
    ImplType  ToRef  Tr  ;
    [T, D, Ra: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [a.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [a.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [a.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigvalsAPI<DeviceBLAS> for (Tr, Homogeneous)
where
    T: BlasFloat,
    D: DimAPI + DimSmallerOneAPI,
    D::SmallerOne: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    type Out = Tensor<Complex<T::Real>, DeviceBLAS, D>;
    #[allow(unused_mut)]
    fn eigvals_f(self) -> Result<Self::Out> {
        let (mut a, _) = self;
        Ok(blas_impl_eigvals_f(ToRef, None, true)?.into_dim::<D>())
    }
}

/* #endregion */

/* #region full-args (generalized) */
#[duplicate_item(
    ImplType  ToRefA  ToRefB  TrA  TrB  ;
    [T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [b.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, DA>]              [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [a.view().to_dyn().into()]                          [b.to_dyn().into()]                                 [&TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [a.view().to_dyn().into()]                          [b.to_dyn().into()]                                 [&TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [a.view().to_dyn().into()]                          [b.view_mut().into_dim::<IxD>().into()]             [&TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [a.to_dyn().into()]                                 [b.view().to_dyn().into()]                          [TensorView<'a, T, DeviceBLAS, DA>]                 [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.to_dyn().into()]                                 [b.view_mut().into_dim::<IxD>().into()]             [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    [T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [a.to_dyn().into()]                                 [b.view().to_dyn().into()]                          [Tensor<T, DeviceBLAS, DA>]                         [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.view_mut().into_dim::<IxD>().into()]             [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [a.view_mut().into_dim::<IxD>().into()]             [b.view().to_dyn().into()]                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.view_mut().into_dim::<IxD>().into()]             [b.to_dyn().into()]                                 [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [a.view_mut().into_dim::<IxD>().into()]             [b.to_dyn().into()]                                 [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.view_mut().into_dim::<IxD>().into()]             [b.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigvalsAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    DA: DimAPI + DimSmallerOneAPI,
    DA::SmallerOne: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    type Out = Tensor<Complex<T::Real>, DeviceBLAS, DA::SmallerOne>;
    #[allow(unused_mut)]
    fn eigvals_f(self) -> Result<Self::Out> {
        let (mut a, mut b) = self;
        Ok(blas_impl_eigvals_f(ToRefA, Some(ToRefB), false)?.into_dim::<DA::SmallerOne>())
    }
}

#[duplicate_item(
    ImplType  ToRefA  ToRefB  TrA  TrB  ;
    [T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [b.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, DA>]              [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [a.view().to_dyn().into()]                          [b.to_dyn().into()]                                 [&TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [a.view().to_dyn().into()]                          [b.to_dyn().into()]                                 [&TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [a.view().to_dyn().into()]                          [b.view_mut().into_dim::<IxD>().into()]             [&TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [a.to_dyn().into()]                                 [b.view().to_dyn().into()]                          [TensorView<'a, T, DeviceBLAS, DA>]                 [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.to_dyn().into()]                                 [b.view_mut().into_dim::<IxD>().into()]             [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    [T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [a.to_dyn().into()]                                 [b.view().to_dyn().into()]                          [Tensor<T, DeviceBLAS, DA>]                         [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [a.to_dyn().into()]                                 [b.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [a.to_dyn().into()]                                 [b.view_mut().into_dim::<IxD>().into()]             [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [a.view_mut().into_dim::<IxD>().into()]             [b.view().to_dyn().into()]                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [&TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.view_mut().into_dim::<IxD>().into()]             [b.to_dyn().into()]                                 [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [a.view_mut().into_dim::<IxD>().into()]             [b.to_dyn().into()]                                 [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [a.view_mut().into_dim::<IxD>().into()]             [b.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigvalsAPI<DeviceBLAS> for (TrA, TrB, Homogeneous)
where
    T: BlasFloat,
    DA: DimAPI + DimSmallerOneAPI,
    DA::SmallerOne: DimAPI,
    DB: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    type Out = Tensor<Complex<T::Real>, DeviceBLAS, DA>;
    #[allow(unused_mut)]
    fn eigvals_f(self) -> Result<Self::Out> {
        let (mut a, mut b, _) = self;
        Ok(blas_impl_eigvals_f(ToRefA, Some(ToRefB), true)?.into_dim::<DA>())
    }
}

/* #endregion */
