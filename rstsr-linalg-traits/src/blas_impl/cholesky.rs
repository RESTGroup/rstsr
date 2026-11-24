use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

/// n-dim `cholesky`: factor every matrix slice of the stack (the two matrix
/// axes are the last two shape axes under a row-major device, the first two
/// under a column-major one; the rest is the batch).
///
/// The whole stack goes through `overwritable_convert` once: an immutable
/// input is factored into a fresh contiguous buffer; a mutable input whose
/// layout matches the device order is factored in place, even when strided
/// (the leading dimension covers the gap); a mutable input whose layout
/// disagrees with the device order is factored in a contiguous work buffer,
/// written back by `clone_to_mut` at the call site.
pub fn blas_impl_cholesky_f<T>(
    a: TensorReference<'_, T, DeviceBLAS, IxD>,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'_, T, DeviceBLAS, IxD>>
where
    T: BlasFloat,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "cholesky: the matrix must be square, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let mut converted = overwritable_convert_with_order(a, order)?;
    let converted_stack = StackLayout::new(converted.view().layout(), 2, order)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at_mut(converted.view_mut(), &converted_stack, i)?.into_dim::<Ix2>();
        let result = ref_impl_cholesky_f(slice.into(), uplo)?;
        result.clone_to_mut();
    }
    Ok(converted)
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        Ok(blas_impl_cholesky_f(a.view().to_dyn().into(), uplo)?.into_owned().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, IxD, IxD> + OpAssignAPI<T, IxD>,
{
    type Out = Tr;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        let result = blas_impl_cholesky_f(a.view_mut().into_dim::<IxD>().into(), uplo)?;
        result.clone_to_mut();
        Ok(a)
    }
}

/* #endregion */

/* #region sub-args */

#[duplicate_item(
    ImplStruct        args_tuple  internal_tuple  ;
    [(Tr, FlagUpLo)] [(a, uplo)] [(a, Some(uplo))];
)]
impl<Tr> CholeskyAPI<DeviceBLAS> for ImplStruct
where
    (Tr, Option<FlagUpLo>): CholeskyAPI<DeviceBLAS>,
{
    type Out = <(Tr, Option<FlagUpLo>) as CholeskyAPI<DeviceBLAS>>::Out;
    fn cholesky_f(self) -> Result<Self::Out> {
        let args_tuple = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f(internal_tuple)
    }
}

#[duplicate_item(
    ImplType                              Tr;
    ['a, T, D, R: DataAPI<Data = Vec<T>>] [&'a TensorAny<R, T, DeviceBLAS, D>];
    ['a, T, D,                          ] [TensorView<'a, T, DeviceBLAS, D>  ];
    [    T, D                           ] [Tensor<T, DeviceBLAS, D>          ];
    ['a, T, D                           ] [TensorMut<'a, T, DeviceBLAS, D>   ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    (Tr, Option<FlagUpLo>): CholeskyAPI<DeviceBLAS>,
{
    type Out = <(Tr, Option<FlagUpLo>) as CholeskyAPI<DeviceBLAS>>::Out;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f((a, None))
    }
}

/* #endregion */
