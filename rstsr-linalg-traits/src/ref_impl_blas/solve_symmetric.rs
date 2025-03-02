use rstsr_blas_traits::prelude::*;
use rstsr_core::error::Result;
use rstsr_core::prelude_dev::*;

pub fn blas_solve_symmetric_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    hermi: bool,
    uplo: FlagUpLo,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat + Send + Sync,
    B: SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_num_threads();
    let result = device.with_num_threads(nthreads, || match hermi {
        false => SYSV::<_, _, false>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
        true => SYSV::<_, _, true>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
    })?;
    Ok(result.1)
}
