//! n-dim `cho_factor` on the BLAS device: the Cholesky factorization of
//! every square matrix slice of the stack (`?POTRF`), in SciPy's `cho_factor`
//! form.
//!
//! Matrix slices: the two matrix axes are the last two shape axes under a
//! row-major device and the first two under a column-major one; the rest is
//! the batch.
//!
//! The result is [`ChoFactorResult`]: `c` of shape `(..., N, N)` holds the
//! factor of each slice in its `uplo` triangle. `uplo` is the requested
//! triangle, and by default the device order's own (`Lower` under a row-major
//! device, `Upper` under a column-major one), so `cho_factor(a).c` and
//! `cholesky(a)` carry the same triangle on one device. The result converts
//! to and from the `(c, uplo)` pair:
//! `let (c, uplo): (_, FlagUpLo) = cho_factor(&a).into();`.
//!
//! The other triangle of `c` is zero for an immutable input (the SciPy
//! default), and keeps the input's values when an owned or mutable input was
//! factored in place. The input policy follows `lu_factor`: an immutable `a`
//! is factored into a fresh buffer, an owned or mutable `a` whose layout is
//! the device order is factored in place and destroyed (strided is fine, the
//! leading dimension covers the gap), and an off-order mutable `a` is
//! factored from a contiguous work buffer that is written back into `a`.
//!
//! A slice that is not positive definite raises the `POTRF` error
//! (`info = i`). Zero-size slices are synthesized without a driver call (the
//! factor is empty, the flag is returned). An input with fewer than two axes,
//! or a non-square slice, is an `InvalidLayout` error.
//!
//! See also [`cholesky`](crate::linalg_auto_impl::cholesky) and
//! [`cho_solve`](crate::linalg_auto_impl::cho_solve).

use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/// Device-facing entry of the n-dim `cho_factor`; see the module
/// documentation for the shapes and the input policy.
#[allow(clippy::type_complexity)]
pub fn blas_impl_cho_factor_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<(TensorMutable<'_, T, DeviceBLAS, IxD>, FlagUpLo)>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let uplo = uplo.unwrap_or(match order {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    // the other triangle is cleaned exactly for an immutable input; the
    // mut-ness is read before the conversion consumes the reference
    let clean = a.is_ref();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "cho_factor: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || -> Result<()> {
        for i in 0..stack.len() {
            let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
            let result = ref_impl_cho_factor_f(slice.into(), uplo, clean)?;
            result.clone_to_mut();
        }
        Ok(())
    };
    device.with_blas_num_threads(nthreads, task)?;
    Ok((converted, uplo))
}

/* #region full-args, read-only input */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> ChoFactorAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = ChoFactorResult<Tensor<T, DeviceBLAS, D>>;
    fn cho_factor_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        let view = a.view().into_dyn();
        let (c, uplo) = blas_impl_cho_factor_f(view.into(), uplo)?;
        Ok(ChoFactorResult { c: c.into_owned().into_dim::<D>(), uplo })
    }
}

/* #endregion */

/* #region full-args, consumed input */

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> ChoFactorAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = ChoFactorResult<Tr>;
    fn cho_factor_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        let view = a.view_mut().into_dim::<IxD>();
        let (result, uplo) = blas_impl_cho_factor_f(view.into(), uplo)?;
        result.clone_to_mut();
        Ok(ChoFactorResult { c: a, uplo })
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct        args_tuple  internal_tuple  ;
   [(Tr, FlagUpLo)] [(a, uplo)] [(a, Some(uplo))];
)]
impl<Tr> ChoFactorAPI<DeviceBLAS> for ImplStruct
where
    (Tr, Option<FlagUpLo>): ChoFactorAPI<DeviceBLAS>,
{
    type Out = <(Tr, Option<FlagUpLo>) as ChoFactorAPI<DeviceBLAS>>::Out;
    fn cho_factor_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        ChoFactorAPI::<DeviceBLAS>::cho_factor_f(internal_tuple)
    }
}

#[duplicate_item(
    ImplType                              Tr;
   ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D>];
   ['a, T, D,                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
   [    T, D                           ] [Tensor<T, DeviceBLAS, D>          ];
   ['a, T, D                           ] [TensorMut<'a, T, DeviceBLAS, D>   ];
)]
impl<ImplType> ChoFactorAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, Option<FlagUpLo>): ChoFactorAPI<DeviceBLAS>,
{
    type Out = <(Tr, Option<FlagUpLo>) as ChoFactorAPI<DeviceBLAS>>::Out;
    fn cho_factor_f(self) -> Result<Self::Out> {
        let a = self;
        ChoFactorAPI::<DeviceBLAS>::cho_factor_f((a, None))
    }
}

/* #endregion */
