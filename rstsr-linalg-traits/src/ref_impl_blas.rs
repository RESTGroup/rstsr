use crate::traits_def::{EighArgs_, PinvResult, SVDArgs_};
use num::{Float, FromPrimitive, Zero};
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
    Ok(result.clone_to_mut())
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
            false => Ok((w, Some(v.clone_to_mut()))),
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
            false => Ok((w, Some(v.clone_to_mut()))),
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
        Ok(a.clone_to_mut())
    };
    device.with_blas_num_threads(nthreads, task)
}

/* #endregion */

/* #region pinv */

pub fn ref_impl_pinv_f<T, B>(
    a: TensorView<T, B, Ix2>,
    atol: Option<T::Real>,
    rtol: Option<T::Real>,
) -> Result<PinvResult<Tensor<T, B, Ix2>>>
where
    T: BlasFloat,
    T::Real: FromPrimitive,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || {
        // compute rcond value
        let atol = atol.unwrap_or(T::Real::zero());
        let rtol = rtol.unwrap_or({
            let [m, n] = *a.shape();
            let mnmax = T::Real::from_usize(m.max(n)).unwrap();
            mnmax * T::Real::epsilon()
        });
        if let (s, Some(u), Some(vt)) = GESDD::default().a(a).full_matrices(false).build()?.run()? {
            let maxs = s.max_all();
            let val = atol + rtol * maxs;
            let rank = s.raw().iter().take_while(|&&x| x > val).count();
            let mut u = u.into_slice((.., ..rank));
            u /= s.i((None, ..rank));
            let a_pinv = GEMM::default()
                .a(vt.i((..rank, ..)).into_reverse_axes().into_dim())
                .b(u.t().into_dim())
                .order(device.default_order())
                .build()?
                .run()?
                .into_owned();
            let pinv = a_pinv.conj();
            Ok(PinvResult { pinv, rank })
        } else {
            unreachable!()
        }
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
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || GESV::default().a(a).b(b).build()?.run();
    let result = device.with_blas_num_threads(nthreads, task)?;
    let (_lu, _piv, x) = result;
    Ok(x.clone_to_mut())
}

/* #endregion */

/* #region solve_symmetric */

pub fn ref_impl_solve_symmetric_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    hermi: bool,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let uplo = uplo.unwrap_or_else(|| match device.default_order() {
        RowMajor => Lower,
        ColMajor => Upper,
    });
    let task = || match hermi {
        false => SYSV::<_, _, false>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
        true => SYSV::<_, _, true>::default().a(a.view()).b(b).uplo(uplo).build()?.run(),
    };
    let result = device.with_blas_num_threads(nthreads, task)?;
    let (_udut, _piv, x) = result;
    Ok(x.clone_to_mut())
}

/* #endregion */

/* #region solve_triangular */

pub fn ref_impl_solve_triangular_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    uplo: Option<FlagUpLo>,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    rstsr_assert!(device.same_device(b.device()), DeviceMismatch)?;
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || TRSM::default().a(a.view()).b(b).uplo(uplo).build()?.run();
    let result = device.with_blas_num_threads(nthreads, task)?;
    Ok(result.clone_to_mut())
}

/* #endregion */

/* #region slogdet */

pub fn ref_impl_slogdet_f<T, B>(a: TensorReference<T, B, Ix2>) -> Result<(T, T::Real)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || {
        let (a, piv) = GETRF::default().a(a).build()?.run()?;
        // pivot indices that may cause sign change
        let mut change_sign = 0;
        let m = piv.shape()[0];
        for i in 0..m {
            if piv[[i]] != i as i32 {
                change_sign += 1;
            }
        }
        // accumulate diagonal values
        let diag_vals = a.view().into_diagonal(());
        let diag_abs = (&diag_vals).abs();
        let diag_sgn = diag_vals.sign();
        let acc_sgn = diag_sgn.prod();
        let acc_sgn = if change_sign % 2 == 0 { acc_sgn } else { -acc_sgn };
        let acc_abs = diag_abs.log().sum();
        Ok((acc_sgn, acc_abs))
    };
    device.with_blas_num_threads(nthreads, task)
}

/* #endregion */

/* #region svd */

pub fn ref_impl_svd_simple_f<'a, T, B>(
    svd_args: SVDArgs_<'a, B, T>,
) -> Result<(Option<Tensor<T, B, Ix2>>, Tensor<T::Real, B, Ix1>, Option<Tensor<T, B, Ix2>>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let SVDArgs_ { a, full_matrices, driver } = svd_args;
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let (full_matrices, compute_uv) = match full_matrices {
        Some(true) => (true, true),
        Some(false) => (false, true),
        None => (false, false),
    };

    let driver = driver.unwrap_or("gesdd");
    match driver {
        "gesvd" => {
            let task = || {
                GESVD::default()
                    .a(a.view())
                    .full_matrices(full_matrices)
                    .compute_uv(compute_uv)
                    .build()?
                    .run()
            };
            let (s, u, vt, _) = device.with_blas_num_threads(nthreads, task)?;
            Ok((u, s, vt))
        },
        "gesdd" => {
            let task = || {
                GESDD::default()
                    .a(a.view())
                    .full_matrices(full_matrices)
                    .compute_uv(compute_uv)
                    .build()?
                    .run()
            };
            let (s, u, vt) = device.with_blas_num_threads(nthreads, task)?;
            Ok((u, s, vt))
        },
        _ => rstsr_invalid!(driver)?,
    }
}

/* #endregion */
