use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, FlagUpLo)
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
        let result = ref_impl_cholesky_f(a.view().into(), Some(uplo))?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

#[duplicate_item(
    ImplType                          Tr                               ;
   [T, D, R: DataAPI<Data = Vec<T>>] [&TensorAny<R, T, DeviceBLAS, D> ];
   [T, D                           ] [TensorView<'_, T, DeviceBLAS, D>];
   [T, D                           ] [TensorCow<'_, T, DeviceBLAS, D> ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        CholeskyAPI::<DeviceBLAS>::cholesky_f((a, uplo))
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for (Tr, FlagUpLo)
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
        let result = ref_impl_cholesky_f(a_ix2.into(), Some(uplo))?;
        result.clone_to_mut();
        Ok(a)
    }
}

#[duplicate_item(
    ImplType   Tr                              ;
   ['a, T, D] [TensorMut<'a, T, DeviceBLAS, D>];
   [    T, D] [Tensor<T, DeviceBLAS, D>       ];
)]
impl<ImplType> CholeskyAPI<DeviceBLAS> for Tr
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tr;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        CholeskyAPI::<DeviceBLAS>::cholesky_f((a, uplo))
    }
}
