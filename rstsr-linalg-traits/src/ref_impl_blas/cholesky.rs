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
        + DeviceCreationTriAPI<T>
        + BlasThreadAPI
        + DeviceRayonAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || POTRF::default().a(a).uplo(uplo).build()?.run();
    let mut result = device.with_blas_num_threads(nthreads, task)?;
    match uplo {
        Upper => triu(result.view_mut()),
        Lower => tril(result.view_mut()),
        _ => rstsr_invalid!(uplo)?,
    };
    Ok(result)
}
