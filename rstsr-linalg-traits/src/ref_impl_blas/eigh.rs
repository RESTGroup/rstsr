use crate::traits::eigh::EighArgs_;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;

pub fn blas_eigh_simple_f<'a, B, T>(
    eigh_args: EighArgs_<'a, '_, B, T>,
) -> Result<(Tensor<T::Real, B, Ix1>, Option<TensorMutable<'a, T, B, Ix2>>)>
where
    T: BlasFloat + Send + Sync,
    B: BlasThreadAPI
        + DeviceRayonAPI
        + DeviceAPI<T, Raw = Vec<T>>
        + DeviceAPI<T::Real, Raw = Vec<T::Real>>
        + DeviceAPI<blas_int, Raw = Vec<blas_int>>
        + DeviceComplexFloatAPI<T, Ix2>
        + DeviceComplexFloatAPI<T::Real, Ix2>
        + DeviceNumAPI<blas_int, Ix1>
        + SYGVDriverAPI<T>
        + SYGVDDriverAPI<T>
        + SYEVDriverAPI<T>
        + SYEVDDriverAPI<T>,
{
    let EighArgs_ {
        a,
        b,
        uplo,
        eigvals_only,
        eig_type,
        subset_by_index: _,
        subset_by_value: _,
        driver,
    } = eigh_args;
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());

    let jobz = if eigvals_only { 'N' } else { 'V' };
    if b.is_some() {
        let driver = driver.unwrap_or("gvd");
        let (w, v) = match driver {
            "gv" => {
                let task = || {
                    SYGV::default()
                        .a(a)
                        .b(b.unwrap())
                        .jobz(jobz)
                        .itype(eig_type)
                        .uplo(uplo)
                        .build()?
                        .run()
                };
                device.with_blas_num_threads(nthreads, task)?
            },
            "gvd" => {
                let task = || {
                    SYGVD::default()
                        .a(a)
                        .b(b.unwrap())
                        .jobz(jobz)
                        .itype(eig_type)
                        .uplo(uplo)
                        .build()?
                        .run()
                };
                device.with_blas_num_threads(nthreads, task)?
            },
            _ => rstsr_invalid!(driver)?,
        };
        match eigvals_only {
            true => Ok((w, None)),
            false => Ok((w, Some(v))),
        }
    } else {
        let driver = driver.unwrap_or("evd");
        let (w, v) = match driver {
            "ev" => {
                let task = || SYEV::default().a(a).jobz(jobz).uplo(uplo).build()?.run();
                device.with_blas_num_threads(nthreads, task)?
            },
            "evd" => {
                let task = || SYEVD::default().a(a).jobz(jobz).uplo(uplo).build()?.run();
                device.with_blas_num_threads(nthreads, task)?
            },
            _ => rstsr_invalid!(driver)?,
        };
        match eigvals_only {
            true => Ok((w, None)),
            false => Ok((w, Some(v))),
        }
    }
}
