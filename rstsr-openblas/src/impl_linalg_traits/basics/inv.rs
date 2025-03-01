use crate::DeviceBLAS;
use rstsr_blas_traits::{
    lapack_solve::{getrf::GETRFDriverAPI, getri::GETRIDriverAPI},
    prelude_dev::BlasFloat,
};
use rstsr_core::prelude_dev::*;
use rstsr_linalg_traits::traits::basics::inv::*;

impl<R, T> LinalgInvAPI<DeviceBLAS> for &TensorAny<R, T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    R: DataAPI<Data = Vec<T>>,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        Ok(blas_inv_f(args.view().into())?.into_owned())
    }
}

impl<T> LinalgInvAPI<DeviceBLAS> for TensorView<'_, T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        Ok(blas_inv_f(args.into())?.into_owned())
    }
}

impl<'a, T> LinalgInvAPI<DeviceBLAS> for TensorMut<'a, T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = TensorMutable<'a, T, DeviceBLAS, Ix2>;
    fn inv_f(args: Self) -> Result<Self::Out> {
        blas_inv_f(args.into())
    }
}

impl<T> LinalgInvAPI<DeviceBLAS> for Tensor<T, DeviceBLAS, Ix2>
where
    T: BlasFloat + Send + Sync,
    DeviceBLAS: DeviceAPI<T, Raw = Vec<T>>
        + GETRIDriverAPI<T>
        + GETRFDriverAPI<T>
        + DeviceComplexFloatAPI<T, Ix2>,
{
    type Out = Tensor<T, DeviceBLAS, Ix2>;
    fn inv_f(mut args: Self) -> Result<Self::Out> {
        Ok(blas_inv_f(args.view_mut().into())?.into_owned())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::DeviceOpenBLAS;
    use rstsr_test_manifest::get_vec;

    #[test]
    fn playground() {
        let device = DeviceOpenBLAS::default();
        let la = [2048, 2048].c();
        let a = Tensor::new(Storage::new(get_vec::<f64>('a').into(), device.clone()), la);
        let a_inv = inv(&a);
        println!("{:?}", a_inv.into_owned());
    }
}
