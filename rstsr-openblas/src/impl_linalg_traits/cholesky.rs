use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude::*;

impl<R, T> LinalgCholeskyAPI<DeviceBLAS> for (&TensorAny<R, T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        Ok(blas_cholesky_f(a.view().into(), uplo)?.into_owned())
    }
}

impl<T> LinalgCholeskyAPI<DeviceBLAS> for (TensorView<'_, T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        Ok(blas_cholesky_f(a.into(), uplo)?.into_owned())
    }
}

impl<'a, T> LinalgCholeskyAPI<DeviceBLAS> for (TensorMut<'a, T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, Ix2>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (a, uplo) = args;
        blas_cholesky_f(a.into(), uplo)
    }
}

impl<T> LinalgCholeskyAPI<DeviceBLAS> for (Tensor<T, DeviceBLAS, Ix2>, FlagUpLo)
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>> + DeviceComplexFloatAPI<T, Ix2> + POTRFDriverAPI<T>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn cholesky_f(args: Self) -> Result<Self::Out> {
        let (mut a, uplo) = args;
        let a_mut = a.view_mut().into();
        blas_cholesky_f(a_mut, uplo)?;
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
    }
}
