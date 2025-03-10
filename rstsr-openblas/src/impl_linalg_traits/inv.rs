use crate::DeviceBLAS;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::prelude_dev::*;

impl<R, T, D> LinalgInvAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    R: DataCloneAPI<Data = Vec<T>>,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            args.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = args.view().into_dim::<Ix2>();
        let result = blas_inv_f(a.into())?.into_owned();
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<T, D> LinalgInvAPI<DeviceBLAS> for TensorView<'_, T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        LinalgInvAPI::<DeviceBLAS>::inv_f(&args)
    }
}

impl<'a, T, D> LinalgInvAPI<DeviceBLAS> for TensorMut<'a, T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, D>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        rstsr_assert_eq!(
            args.ndim(),
            2,
            InvalidLayout,
            "Currently we can only handle 2-D matrix."
        )?;
        let a = args.into_dim::<Ix2>();
        let result = blas_inv_f(a.into())?;
        Ok(result.into_dim::<IxD>().into_dim::<D>())
    }
}

impl<T, D> LinalgInvAPI<DeviceBLAS> for Tensor<T, DeviceBLAS, D>
where
    T: BlasFloat + Send + Sync,
    D: DimAPI,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, D>;
    fn inv_f(mut args: Self) -> Result<Self::Out> {
        LinalgInvAPI::<DeviceBLAS>::inv_f(args.view_mut())?;
        Ok(args)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceBLAS;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceBLAS::default();
        let la = [2048, 2048].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let a_inv = inv(&a);
        println!("{:?}", a_inv.into_owned());
    }
}
