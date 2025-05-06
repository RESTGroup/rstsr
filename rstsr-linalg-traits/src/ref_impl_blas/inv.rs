use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;

pub fn blas_inv_f<T, B>(a: TensorReference<T, B, Ix2>) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat,
    B: GETRFDriverAPI<T>
        + GETRIDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + DeviceRayonAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || {
        let (mut a, ipiv) = GETRF::default().a(a).build()?.run()?;
        GETRI::default().a(a.view_mut()).ipiv(ipiv.view()).build()?.run()?;
        Ok(a)
    };
    device.with_blas_num_threads(nthreads, task)
}
