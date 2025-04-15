use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<Ra, Rb, T, D> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<Ra, T, DeviceBLAS, D>, &TensorAny<Rb, T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    Ra: DataCloneAPI<Data = Vec<T>>,
    Rb: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view().into_dim::<Ix2>();
        let result = blas_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        return Ok(result.into_owned().into_dim::<IxD>().into_dim::<D>());
    }
}

impl<T, D> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, TensorView<'_, T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        LinalgSolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f((&a, &b, hermi, uplo))
    }
}

impl<R, T, D> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (&TensorAny<R, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, mut b, hermi, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        blas_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
        Ok(b)
    }
}

impl<T, D> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (TensorView<'_, T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (a, b, hermi, uplo) = args;
        LinalgSolveSymmetricAPI::<DeviceBLAS>::solve_symmetric_f((&a, b, hermi, uplo))
    }
}

impl<T, D> LinalgSolveSymmetricAPI<DeviceBLAS>
    for (Tensor<T, DeviceBLAS, D>, Tensor<T, DeviceBLAS, D>, bool, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn solve_symmetric_f(args: Self) -> Result<Self::Out> {
        let (mut a, mut b, hermi, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        rstsr_assert_eq!(b.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a_view = a.view_mut().into_dim::<Ix2>();
        let b_view = b.view_mut().into_dim::<Ix2>();
        blas_solve_symmetric_f(a_view.into(), b_view.into(), hermi, uplo)?;
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
        assert_eq!(x.as_ptr(), ptr_b);
    }
}
