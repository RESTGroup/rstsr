use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

/* #region full-args */

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, Option<FlagUpLo>)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = ref_impl_cholesky_f(a.view().into(), uplo)?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
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
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tr;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_ix2 = a.view_mut().into_dim::<Ix2>();
        let result = ref_impl_cholesky_f(a_ix2.into(), uplo)?;
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
