use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<Ra, Rb, T> LinalgSolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, Ix2>, &TensorAny<Rb, T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        Ok(blas_solve_triangular_f(a.view().into(), b.view().into(), uplo)?.into_owned())
    }
}

impl<T> LinalgSolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, Ix2>, TensorView<'_, T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        solve_triangular_f((&a, &b, uplo))
    }
}

impl<R, T> LinalgSolveTriangularAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, mut b, uplo) = args;
        blas_solve_triangular_f(a.view().into(), b.view_mut().into(), uplo)?;
        Ok(b)
    }
}

impl<T> LinalgSolveTriangularAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (a, b, uplo) = args;
        solve_triangular_f((&a, b, uplo))
    }
}

impl<T> LinalgSolveTriangularAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + TRSMDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_triangular_f(args: Self) -> Result<Self::Out> {
        let (mut a, mut b, uplo) = args;
        blas_solve_triangular_f(a.view_mut().into(), b.view_mut().into(), uplo)?;
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
