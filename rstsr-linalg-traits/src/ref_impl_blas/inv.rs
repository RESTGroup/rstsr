use rstsr_blas_traits::prelude::*;
use rstsr_core::error::Result;
use rstsr_core::prelude_dev::*;

pub fn blas_inv_f<T, B>(a: TensorReference<T, B, Ix2>) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat + Send + Sync,
    B: GETRFDriverAPI<T>
        + GETRIDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_num_threads();
    device.with_num_threads(nthreads, || {
        let (mut a, ipiv) = GETRF::default().a(a).build()?.run()?;
        GETRI::default().a(a.view_mut()).ipiv(ipiv.view()).build()?.run()?;
        Ok(a)
    })
}
