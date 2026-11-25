//! n-dim `eig` on the BLAS device: eigenvalues and the requested eigenvector
//! blocks of every square matrix slice, standard or generalized `A v = λ B v`.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch. The batch dims of `a` and `b` broadcast against each other;
//! under a column-major device they align from the left, so a `b` of shape
//! `(M, M, k)` pairs with the first batch axis of `a`.
//!
//! The spectrum is always complex (`Complex<T::Real>`), in LAPACK's order and
//! unsorted; the result is [`EigResult`] with `Some` exactly for the requested
//! [`EigSide`] (the bare form is `Right`). `w` has shape `(..., M)`, or
//! `(..., 2, M)` holding `(alpha, beta)` when the [`Homogeneous`] marker is
//! given (`beta` is identically `1` for the standard problem; the generalized
//! eigenvalues are `alpha / beta`).
//!
//! This follows SciPy's `eig`, except that the result is complex in every
//! case: SciPy and NumPy return real-typed eigenvalues and eigenvectors when a
//! real input has a real spectrum. Standard eigenvectors are `GEEV`'s (unit
//! 2-norm, largest component real); generalized ones are rescaled to unit
//! 2-norm on both sides. A vanishing `beta` gives `inf` (`nan` for
//! `alpha = beta = 0`; its form is decided per matrix slice).
//!
//! Overloads (positional order `a, b, side, homogeneous`; every matrix slot
//! takes `&TensorAny`, `TensorView`, `Tensor` or `TensorMut`):
//!
//! - `eig(a)`, `eig((a, EigSide))`, `eig((a, "left"))`, `eig((a, FlagSide::L))`
//! - `eig((a, Homogeneous))`, `eig((a, EigSide, Homogeneous))`, ...
//! - the same eight forms with `b` after `a`: `eig((a, b))`, ...
//!
//! The static-dim overloads type the result by `a`: a `b` with more batch
//! axes than `a` is rejected there with `InvalidLayout` (use dynamic dims for
//! that broadcast).
//!
//! The inputs are consumed as scratch: an in-order owned or mutable input is
//! destroyed by the per-slice driver call when its batch needs no broadcast
//! (`GEEV`/`GGEV` overwrite it; nothing is written back); every other input
//! is copied into the call order.
//!
//! See also [`eigvals`](crate::linalg_auto_impl::eigvals).

use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `eig`; see [`ref_impl_eig_nd_f`] for the
/// semantics of the stack walk and the output rules.
#[allow(clippy::type_complexity)]
pub fn blas_impl_eig_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    b: Option<TensorReference<'_, T, DeviceBLAS, IxD>>,
    side: EigSide,
    homogeneous: bool,
) -> Result<
    EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, IxD>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, IxD>>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, IxD>>,
    >,
>
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
    let (eigenvalues, left, right) = ref_impl_eig_nd_f(a, b, side, homogeneous, "eig")?;
    Ok(EigResult { eigenvalues, left, right })
}

/* #region full-args */

/// (Tr, EigSide) and (Tr, EigSide, Homogeneous): the standard-problem cores
#[duplicate_item(
    ImplType                                   ToRef                                        Tr                               ;
    [T, D, Ra: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [a.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [a.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [a.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, EigSide)
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
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D::SmallerOne>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, D>>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, D>>,
    >;
    #[allow(unused_mut)]
    fn eig_f(self) -> Result<Self::Out> {
        let (mut a, side) = self;
        let EigResult { eigenvalues, left, right } = blas_impl_eig_f(ToRef, None, side, false)?;
        Ok(EigResult {
            eigenvalues: eigenvalues.into_dim::<D::SmallerOne>(),
            left: left.map(|v| v.into_dim::<D>()),
            right: right.map(|v| v.into_dim::<D>()),
        })
    }
}

#[duplicate_item(
    ImplType                                   ToRef                                        Tr                               ;
    [T, D, Ra: DataAPI<Data = Vec<T>>]  [a.view().to_dyn().into()]                          [&TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [a.to_dyn().into()]                                 [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [a.to_dyn().into()]                                 [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [a.view_mut().into_dim::<IxD>().into()]             [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, EigSide, Homogeneous)
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
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, D>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, D>>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, D>>,
    >;
    #[allow(unused_mut)]
    fn eig_f(self) -> Result<Self::Out> {
        let (mut a, side, _) = self;
        let EigResult { eigenvalues, left, right } = blas_impl_eig_f(ToRef, None, side, true)?;
        Ok(EigResult {
            eigenvalues: eigenvalues.into_dim::<D>(),
            left: left.map(|v| v.into_dim::<D>()),
            right: right.map(|v| v.into_dim::<D>()),
        })
    }
}

/* #endregion */

/* #region sub-args */
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let a = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::Right))
    }
}
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, &str)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, side) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::try_from(side)?))
    }
}
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, FlagSide)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, side) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::from(side)))
    }
}
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, Homogeneous)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::Right, Homogeneous))
    }
}
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, &str, Homogeneous)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, side, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::try_from(side)?, Homogeneous))
    }
}
#[duplicate_item(
    ImplType  Tr  ;
    ['a, T, D, Ra: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorView<'a, T, DeviceBLAS, D>];
    [T, D]                                  [Tensor<T, DeviceBLAS, D>];
    ['a, T, D]                              [TensorMut<'a, T, DeviceBLAS, D>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (Tr, FlagSide, Homogeneous)
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(Tr, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, side, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, EigSide::from(side), Homogeneous))
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
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, EigSide)
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
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, DA::SmallerOne>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, DA>>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, DA>>,
    >;
    #[allow(unused_mut)]
    fn eig_f(self) -> Result<Self::Out> {
        let (mut a, mut b, side) = self;
        check_static_generalized_ranks(a.ndim(), b.ndim(), a.layout().shape().const_ndim(), "eig")?;
        let EigResult { eigenvalues, left, right } = blas_impl_eig_f(ToRefA, Some(ToRefB), side, false)?;
        Ok(EigResult {
            eigenvalues: eigenvalues.into_dim::<DA::SmallerOne>(),
            left: left.map(|v| v.into_dim::<DA>()),
            right: right.map(|v| v.into_dim::<DA>()),
        })
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
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, EigSide, Homogeneous)
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
    type Out = EigResult<
        Tensor<Complex<T::Real>, DeviceBLAS, DA>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, DA>>,
        Option<Tensor<Complex<T::Real>, DeviceBLAS, DA>>,
    >;
    #[allow(unused_mut)]
    fn eig_f(self) -> Result<Self::Out> {
        let (mut a, mut b, side, _) = self;
        check_static_generalized_ranks(a.ndim(), b.ndim(), a.layout().shape().const_ndim(), "eig")?;
        let EigResult { eigenvalues, left, right } = blas_impl_eig_f(ToRefA, Some(ToRefB), side, true)?;
        Ok(EigResult {
            eigenvalues: eigenvalues.into_dim::<DA>(),
            left: left.map(|v| v.into_dim::<DA>()),
            right: right.map(|v| v.into_dim::<DA>()),
        })
    }
}

/* #endregion */

/* #region sub-args (generalized) */
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::Right))
    }
}
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, &str)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, side) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::try_from(side)?))
    }
}
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, FlagSide)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, side) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::from(side)))
    }
}
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, Homogeneous)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::Right, Homogeneous))
    }
}
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, &str, Homogeneous)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, side, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::try_from(side)?, Homogeneous))
    }
}
#[duplicate_item(
    ImplType  TrA  TrB  ;
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB, Rb: DataAPI<Data = Vec<T>>]  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                                  [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, Ra: DataAPI<Data = Vec<T>>, DB]                              [&'a TensorAny<Ra, T, DeviceBLAS, DA>]              [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorView<'a, T, DeviceBLAS, DA>]                 [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorView<'a, T, DeviceBLAS, DA>]                 [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorView<'a, T, DeviceBLAS, DA>]                 [TensorMut<'b, T, DeviceBLAS, DB>];
    ['b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                                  [Tensor<T, DeviceBLAS, DA>]                         [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorView<'b, T, DeviceBLAS, DB>];
    [T, DA, DB]                                                                  [Tensor<T, DeviceBLAS, DA>]                         [Tensor<T, DeviceBLAS, DB>];
    ['b, T, DA, DB]                                                              [Tensor<T, DeviceBLAS, DA>]                         [TensorMut<'b, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB, Rb: DataAPI<Data = Vec<T>>]                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [&'b TensorAny<Rb, T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorView<'b, T, DeviceBLAS, DB>];
    ['a, T, DA, DB]                                                              [TensorMut<'a, T, DeviceBLAS, DA>]                  [Tensor<T, DeviceBLAS, DB>];
    ['a, 'b, T, DA, DB]                                                          [TensorMut<'a, T, DeviceBLAS, DA>]                  [TensorMut<'b, T, DeviceBLAS, DB>];
)]
impl<ImplType> EigAPI<DeviceBLAS> for (TrA, TrB, FlagSide, Homogeneous)
where
    T: BlasFloat,
    DA: DimAPI,
    DB: DimAPI,
    (TrA, TrB, EigSide, Homogeneous): EigAPI<DeviceBLAS>,
{
    type Out = <(TrA, TrB, EigSide, Homogeneous) as EigAPI<DeviceBLAS>>::Out;
    fn eig_f(self) -> Result<Self::Out> {
        let (a, b, side, _) = self;
        EigAPI::<DeviceBLAS>::eig_f((a, b, EigSide::from(side), Homogeneous))
    }
}

/* #endregion */
