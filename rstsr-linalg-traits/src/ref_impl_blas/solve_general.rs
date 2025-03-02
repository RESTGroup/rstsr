use rstsr_blas_traits::prelude::*;
use rstsr_core::error::Result;
use rstsr_core::prelude_dev::*;

pub fn blas_solve_general_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat + Send + Sync,
    B: GESVDriverAPI<T>
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<blasint, Raw = Vec<blasint>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceNumAPI<blasint, Ix1>
        + BlasThreadAPI,
{
    let device = a.device().clone();
    let nthreads = device.get_num_threads();
    let result = device.with_num_threads(nthreads, || GESV::default().a(a).b(b).build()?.run())?;
    Ok(result.1)
}
