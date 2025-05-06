use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;

pub fn blas_solve_symmetric_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    hermi: bool,
    uplo: FlagUpLo,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: SYSVDriverAPI<T, false>
        + SYSVDriverAPI<T, true>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + DeviceRayonAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || match hermi {
        false => SYSV::<_, _, false>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
        true => SYSV::<_, _, true>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
    };
    let result = device.with_blas_num_threads(nthreads, task)?;
    Ok(result.1)
}
