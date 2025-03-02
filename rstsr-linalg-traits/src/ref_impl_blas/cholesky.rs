use rstsr_blas_traits::prelude::*;
use rstsr_core::error::Result;
use rstsr_core::prelude_dev::*;

pub fn blas_cholesky_f<T, B>(
    a: TensorReference<T, B, Ix2>,
    uplo: FlagUpLo,
) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat + Send + Sync,
    B: POTRFDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceComplexFloatAPI<T, Ix2>
        + BlasThreadAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_num_threads();
    device.with_num_threads(nthreads, || {
        let a = POTRF::default().a(a).uplo(uplo).build()?.run()?;
        Ok(a)
    })
}
