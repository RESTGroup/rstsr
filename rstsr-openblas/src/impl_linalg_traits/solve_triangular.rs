use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<Ra, Rb, T, D> LinalgSolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    Ra: DataCloneAPI<Data = Vec<T>>,
    Rb: DataCloneAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = blas_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        return Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>());
    }
}

impl<T, D> LinalgSolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        LinalgSolveTriangularAPI::<DeviceBLAS>::solve_triangular_f((&a, &b, uplo))
    }
}

impl<R, T, D> LinalgSolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, mut b, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        blas_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        Ok(b)
    }
}

impl<T, D> LinalgSolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        LinalgSolveTriangularAPI::<DeviceBLAS>::solve_triangular_f((&a, b, uplo))
    }
}

impl<T, D> LinalgSolveTriangularAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (mut a, mut b, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        blas_solve_triangular_f(a_view.into(), b_view.into(), uplo)?;
        Ok(b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_solve_triangular() {
        let device = DeviceBLAS::default();
        let vec_a = vec![1.0, 2.0, 3.0, 4.0];
        let vec_b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = asarray((vec_a, [2, 2].c(), &device)).into_dim::<Ix2>();
        let b = asarray((vec_b, [2, 3].c(), &device)).into_dim::<Ix2>();
        let ptr_b = b.as_ptr();
        let x = solve_triangular((&a, &b, Lower));
        println!("{:?}", x);

        let x = solve_triangular((&a, b, Upper));
        println!("{:?}", x);
        assert!(x.as_ptr() == ptr_b);
    }
}
