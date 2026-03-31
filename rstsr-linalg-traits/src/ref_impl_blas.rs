use crate::traits_def::{EighArgs_, PinvResult, SVDArgs_};
use num::{Float, FromPrimitive, Zero};
use rstsr_blas_traits::prelude::*;
use rstsr_core::prelude::rt;
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
    let EighArgs_ { a, b, uplo, eigvals_only, eig_type, subset_by_index: _, subset_by_value: _, driver } = eigh_args;
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());

    let jobz = if eigvals_only { 'N' } else { 'V' };
    if b.is_some() {
        let driver = driver.unwrap_or("gvd");
        let (w, v) = match driver {
            "gv" => {
                let task = || SYGV::default().a(a).b(b.unwrap()).jobz(jobz).itype(eig_type).uplo(uplo).build()?.run();
                device.with_blas_num_threads(nthreads, task)?
            },
            "gvd" => {
                let task = || SYGVD::default().a(a).b(b.unwrap()).jobz(jobz).itype(eig_type).uplo(uplo).build()?.run();
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
            if piv[[i]] != i as blas_int {
                change_sign += 1;
            }
        }
        // accumulate diagonal values
        let diag_vals = a.view().into_diagonal(());
        let diag_abs = (&diag_vals).abs();
        let diag_sgn = diag_vals.sign();
        let acc_sgn = diag_sgn.prod();
        let acc_sgn = if change_sign % 2 == 0 { acc_sgn } else { -acc_sgn };
        let acc_abs = rt::log(diag_abs).sum();
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
            let task =
                || GESVD::default().a(a.view()).full_matrices(full_matrices).compute_uv(compute_uv).build()?.run();
            let (s, u, vt, _) = device.with_blas_num_threads(nthreads, task)?;
            Ok((u, s, vt))
        },
        "gesdd" => {
            let task =
                || GESDD::default().a(a.view()).full_matrices(full_matrices).compute_uv(compute_uv).build()?.run();
            let (s, u, vt) = device.with_blas_num_threads(nthreads, task)?;
            Ok((u, s, vt))
        },
        _ => rstsr_invalid!(driver)?,
    }
}

/* #endregion */

/* #region qr */

use crate::traits_def::QRResult;

/// Reference implementation of QR decomposition
///
/// # Arguments
/// * `a` - Input matrix
/// * `mode` - "reduced" (or "economic"), "complete", "r", or "raw"
/// * `pivoting` - Whether to use column pivoting (GEQP3 instead of GEQRF)
///
/// # Returns
/// Unified QRResult with fields populated based on mode:
/// - "reduced" (or "economic"): q=Some(Q with shape [m,k]), r=Some(R with shape [k,n])
/// - "complete": q=Some(Q with shape [m,m]), r=Some(R with shape [m,n])
/// - "r": q=None, r=Some(R with shape [k,n])
/// - "raw": q=None, r=None, h=Some(packed), tau=Some(tau)
///
/// Note: "economic" is an alias for "reduced" (SciPy vs NumPy terminology).
pub fn ref_impl_qr_f<'a, T, B>(
    a: TensorReference<'a, T, B, Ix2>,
    mode: &'static str,
    pivoting: bool,
) -> Result<
    QRResult<
        Tensor<T, B, Ix2>,
        Tensor<T, B, Ix2>,
        TensorMutable<'a, T, B, Ix2>,
        Tensor<T, B, Ix1>,
        Tensor<blas_int, B, Ix1>,
    >,
>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());

    // Normalize mode: "economic" is an alias for "reduced" (SciPy vs NumPy terminology)
    let mode = if mode == "economic" { "reduced" } else { mode };

    rstsr_assert!(
        matches!(mode, "reduced" | "complete" | "r" | "raw"),
        InvalidValue,
        "mode must be 'reduced' (or 'economic'), 'complete', 'r', or 'raw'"
    )?;

    let task = || {
        let [m, n] = *a.view().shape();
        let k = m.min(n);

        // Step 1: Perform QR factorization (GEQRF or GEQP3)
        let (qr, tau, p) = if pivoting {
            let (qr, jpvt, tau) = GEQP3::default().a(a).build()?.run()?;
            (qr, tau, Some(jpvt))
        } else {
            let (qr, tau) = GEQRF::default().a(a).build()?.run()?;
            (qr, tau, None)
        };

        // Handle 'raw' mode: return packed QR and tau directly
        if mode == "raw" {
            return Ok(QRResult { q: None, r: None, h: Some(qr), tau: Some(tau), p });
        }

        // Step 2: Extract R (upper triangular part)
        // R has shape (k, n) where k = min(m, n)
        let r = {
            let qr_view = qr.view();
            let mut r: Tensor<T, B, Ix2> = zeros_f(([k, n].c(), &device))?.into_dim::<Ix2>();
            // Extract upper triangular part: R[i,j] = QR[i,j] for i <= j, else 0
            // manually copy by strides and shapes
            let r_stride = *r.stride();
            let qr_stride = *qr_view.stride();
            let r_offset = r.offset();
            let qr_offset = qr_view.offset();
            let vec_r = r.raw_mut();
            let vec_qr = qr_view.raw();
            for i in 0..(k as isize) {
                for j in i..(n as isize) {
                    let idx_r = (i * r_stride[0] + j * r_stride[1] + r_offset as isize) as usize;
                    let idx_qr = (i * qr_stride[0] + j * qr_stride[1] + qr_offset as isize) as usize;
                    vec_r[idx_r] = vec_qr[idx_qr];
                }
            }
            r
        };

        // Handle 'r' mode: return only R
        if mode == "r" {
            return Ok(QRResult { q: None, r: Some(r), h: None, tau: None, p });
        }

        // Step 3: Generate Q using ORGQR
        let q = if mode == "reduced" || m <= n {
            // For 'reduced' mode: generate Q with k columns
            // For 'complete' mode with m <= n: k == m, so same as reduced
            // ORGQR requires m >= n (columns of output Q)
            if m < n {
                // For M < N: QR has shape [M, N], but ORGQR requires m >= n
                // We need to extract the first M columns of QR for ORGQR
                // The Householder vectors are stored in the first min(M,N)=M columns
                let mut qr_mxm = unsafe { empty_f(([m, m].c(), &device))?.into_dim::<Ix2>() };
                qr_mxm.view_mut().assign(&qr.view().i((.., ..m)));
                ORGQR::default().a(qr_mxm.view_mut()).tau(tau.view()).build()?.run()?.into_owned()
            } else {
                ORGQR::default().a(qr.view()).tau(tau.view()).build()?.run()?.into_owned()
            }
        } else {
            // For 'complete' mode with m > n:
            // We need to generate an m x m orthogonal matrix.
            // ORGQR generates Q from the first n columns of the input matrix.
            // We create an m x m matrix with identity extension and use ORGQR.
            let mut q_full = zeros_f(([m, m].c(), &device))?.into_dim::<Ix2>();
            let qr_view = qr.view();
            // Copy QR into first n columns
            q_full.i_mut((.., ..n)).assign(&qr_view);
            // Fill remaining columns with identity matrix
            for j in n..m {
                q_full[[j, j]] = T::one();
            }
            // Generate Q using ORGQR with k=n (use all n Householder vectors)
            ORGQR::default().a(q_full.view_mut()).tau(tau.view()).build()?.run()?.into_owned()
        };

        Ok(QRResult { q: Some(q), r: Some(r), h: None, tau: None, p })
    };

    device.with_blas_num_threads(nthreads, task)
}

/* #endregion */
