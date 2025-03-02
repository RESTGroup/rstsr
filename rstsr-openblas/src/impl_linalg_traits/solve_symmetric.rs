use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude::*;

impl<Ra, Rb, T> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, Ix2>, &TensorAny<Rb, T, DeviceBLAS, Ix2>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    Ra: DataAPI<Data = Vec<T>>,
    Rb: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        Ok(blas_solve_symmetric_f(a.view().into(), b.view().into(), hermi, uplo)?.into_owned())
    }
}

impl<T> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, Ix2>, TensorView<'_, T, DeviceBLAS, Ix2>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        solve_symmetric_f((&a, &b, hermi, uplo))
    }
}

impl<R, T> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, mut b, hermi, uplo) = args;
        blas_solve_symmetric_f(a.view().into(), b.view_mut().into(), hermi, uplo)?;
        Ok(b)
    }
}

impl<T> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        solve_symmetric_f((&a, b, hermi, uplo))
    }
}

impl<T> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, Ix2>, Tensor<T, DeviceBLAS, Ix2>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (mut a, mut b, hermi, uplo) = args;
        blas_solve_symmetric_f(a.view_mut().into(), b.view_mut().into(), hermi, uplo)?;
        Ok(b)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_solve_symmetric() {
        let device = DeviceBLAS::default();
        let vec_a = vec![1.0, 2.0, 3.0, 3.0];
        let vec_b = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let a = asarray((vec_a, [2, 2].c(), &device)).into_dim::<Ix2>();
        let b = asarray((vec_b, [2, 3].c(), &device)).into_dim::<Ix2>();
        let ptr_b = b.as_ptr();
        let x = solve_symmetric((&a, &b, false, Lower));
        println!("{:?}", x);

        let x = solve_symmetric((&a, b, false, Upper));
        println!("{:?}", x);
        assert!(x.as_ptr() == ptr_b);
    }
}
