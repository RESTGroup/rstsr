use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<Ra, Rb, T, D> SolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        return Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>());
    }
}

impl<T, D> SolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        SolveTriangularAPI::<DeviceBLAS>::solve_triangular_f((&a, &b, uplo))
    }
}

impl<R, T, D> SolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    R: DataAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        Ok(b)
    }
}

impl<T, D> SolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (a, b, uplo) = self;
        SolveTriangularAPI::<DeviceBLAS>::solve_triangular_f((&a, b, uplo))
    }
}

impl<T, D> SolveTriangularAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat,
    D: DimAPI,
    DeviceBLAS: LapackDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(self) -> Result<Self::Out> {
        let (mut a, mut b, uplo) = self;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        ref_impl_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        Ok(b)
    }
}
