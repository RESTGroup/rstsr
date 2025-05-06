use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;

pub fn blas_solve_triangular_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    uplo: FlagUpLo,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: TRSMDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + BlasThreadAPI
        + DeviceRayonAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || TRSM::default().a(a.view()).b(b).uplo(uplo).build()?.run();
    let result = device.with_blas_num_threads(nthreads, task)?;
    Ok(result)
}
