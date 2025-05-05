use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> LinalgCholeskyAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.view().into_dim::<Ix2>();
        let result = blas_cholesky_f(a.view().into(), uplo)?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<T, D> LinalgCholeskyAPI<DeviceBLAS> for (TensorView<'_, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        LinalgCholeskyAPI::<DeviceBLAS>::cholesky_f((&a, uplo))
    }
}

impl<'a, T, D> LinalgCholeskyAPI<DeviceBLAS> for (TensorMut<'a, T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, D>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        rstsr_assert_eq!(a.ndim(), 2, InvalidLayout, "Currently we can only handle 2-D matrix.")?;
        let a = a.into_dim::<Ix2>();
        let result = blas_cholesky_f(a.into(), uplo)?;
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<T, D> LinalgCholeskyAPI<DeviceBLAS> for (Tensor<T, DeviceBLAS, D>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (mut a, uplo) = args;
        LinalgCholeskyAPI::<DeviceBLAS>::cholesky_f((a.view_mut(), uplo))?;
        Ok(a)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceBLAS;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
        let vec_a = [1, 1, 2, 1, 3, 1, 2, 1, 8].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let a = asarray((vec_a, [3, 3].c(), &device)).into_dim::<Ix2>();
        let a_cholesky = cholesky((a, Lower));
        println!("{:?}", a_cholesky);
        let vec_a = [1, 1, 2, 1, 3, 1, 2, 1, 8].iter().map(|&x| x as f64).collect::<Vec<_>>();
        let a = asarray((vec_a, vec![3, 3].c(), &device));
        let a_cholesky = cholesky((a, Upper));
        println!("{:?}", a_cholesky);
    }
}
