use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<Ra, Rb, T, D> SolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        return Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>());
    }
}

impl<T, D> SolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = self;
        SolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f((&a, &b, hermi, uplo))
    }
}

impl<R, T, D> SolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, mut b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        Ok(b)
    }
}

impl<T, D> SolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = self;
        SolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f((&a, b, hermi, uplo))
    }
}

impl<T, D> SolveSymmetricAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(self) -> Result<Self::Out> {
        let (mut a, mut b, hermi, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        ref_impl_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        Ok(b)
    }
}
