use crate::DeviceBLAS;
use duplicate::duplicate_item;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> CholeskyAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
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

impl<R, T, D> CholeskyAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
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
    TSR;
    [TensorView<'_, T, DeviceBLAS, D>];
    [TensorCow<'_, T, DeviceBLAS, D>]
)]
impl<T, D> CholeskyAPI<DeviceBLAS> for (TSR, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f((&a, uplo))
    }
}

#[duplicate_item(
    TSR;
    [TensorView<'_, T, DeviceBLAS, D>];
    [TensorCow<'_, T, DeviceBLAS, D>]
)]
impl<T, D> CholeskyAPI<DeviceBLAS> for TSR
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f(&a)
    }
}

impl<'a, T, D> CholeskyAPI<DeviceBLAS> for (TensorMut<'a, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (a, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.into_dim::<Ix2>();
        let result = ref_impl_cholesky_f(a.into(), Some(uplo))?;
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<'a, T, D> CholeskyAPI<DeviceBLAS> for TensorMut<'a, T, DeviceBLAS, D>
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let a = self;
        let uplo = match a.device().default_order() {
            RowMajor => Lower,
            ColMajor => Upper,
        };
        CholeskyAPI::<DeviceBLAS>::cholesky_f((a, uplo))
    }
}

impl<T, D> CholeskyAPI<DeviceBLAS> for (Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let (mut a, uplo) = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f((a.view_mut(), uplo))?;
        Ok(a)
    }
}

impl<T, D> CholeskyAPI<DeviceBLAS> for Tensor<T, DeviceBLAS, D>
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(self) -> Result<Self::Out> {
        let mut a = self;
        CholeskyAPI::<DeviceBLAS>::cholesky_f(a.view_mut())?;
        Ok(a)
    }
}
