use crate::traits_def::EighArgs_;
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude_dev::*;

/* #region cholesky */

pub fn ref_impl_cholesky_f<T, B>(
    a: TensorReference<T, B, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let uplo = uplo.unwrap_or_else(|| match device.default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let task = || POTRF::default().a(a).uplo(uplo).build()?.run();
    let mut result = device.with_blas_num_threads(nthreads, task)?;
    match uplo {
        Upper => triu(result.view_mut()),
        Lower => tril(result.view_mut()),
    };
    Ok(result)
}

/* #endregion */

/* #region eigh */

pub fn ref_impl_eigh_simple_f<'a, B, T>(
    eigh_args: EighArgs_<'a, '_, B, T>,
) -> Result<(Tensor<T::Real, B, Ix1>, Option<TensorMutable<'a, T, B, Ix2>>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
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

/* #endregion */

/* #region inv */

pub fn ref_impl_inv_f<T, B>(a: TensorReference<T, B, Ix2>) -> Result<TensorMutable<T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
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

/* #endregion */

/* #region solve_general */

pub fn ref_impl_solve_general_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || GESV::default().a(a).b(b).build()?.run();
    let result = device.with_blas_num_threads(nthreads, task)?;
    let (_lu, _piv, x) = result;
    Ok(x)
}

/* #endregion */

/* #region solve_symmetric */

pub fn ref_impl_solve_symmetric_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    hermi: bool,
    uplo: FlagUpLo,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || match hermi {
        false => SYSV::<_, _, false>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
        true => SYSV::<_, _, true>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
    };
    let result = device.with_blas_num_threads(nthreads, task)?;
    let (_udut, _piv, x) = result;
    Ok(x)
}

/* #endregion */

/* #region solve_triangular */

pub fn ref_impl_solve_triangular_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    uplo: FlagUpLo,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || TRSM::default().a(a.view()).b(b).uplo(uplo).build()?.run();
    let result = device.with_blas_num_threads(nthreads, task)?;
    Ok(result)
}

/* #endregion */
