use crate::stack::{stack_shape, stack_slice_at, stack_slice_at_mut, StackLayout};
use crate::traits_def::{EigSide, EighArgs_, PinvResult, QrMode, SVDArgs_};
use num::{Complex, Float, FromPrimitive, One, Zero};
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

/* #region eig */

/// Eigenvalues (and the requested eigenvector blocks) of one 2-D matrix; the
/// input is consumed as scratch (destroyed by `GEEV`).
pub fn ref_impl_eig_simple_f<T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    left: bool,
    right: bool,
) -> Result<(
    Tensor<Complex<T::Real>, B, Ix1>,
    Option<Tensor<Complex<T::Real>, B, Ix2>>,
    Option<Tensor<Complex<T::Real>, B, Ix2>>,
)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    device.with_blas_num_threads(nthreads, || GEEV::default().a(a).left(left).right(right).build()?.run())
}

/// Rescale every column of `v` to unit 2-norm (SciPy's convention for `GGEV`;
/// LAPACK scales the largest component to `abs(re) + abs(im) = 1`).
fn normalize_eigvec_columns<R, B>(mut v: Tensor<Complex<R>, B, Ix2>) -> Tensor<Complex<R>, B, Ix2>
where
    R: Float,
    B: DeviceAPI<Complex<R>, Raw = Vec<Complex<R>>>,
{
    let (m, n) = (v.nrow(), v.ncol());
    let offset = v.offset();
    let (rs, cs) = (v.stride()[0], v.stride()[1]);
    let raw = v.raw_mut();
    for j in 0..n {
        let col = |i: usize| (offset as isize + i as isize * rs + j as isize * cs) as usize;
        let norm_sq = (0..m).map(|i| raw[col(i)].norm_sqr()).fold(R::zero(), |acc, x| acc + x);
        let norm = norm_sq.sqrt();
        for i in 0..m {
            raw[col(i)] = raw[col(i)] / norm;
        }
    }
    v
}

/// Generalized eigenvalues of `A v = lambda B v` (and the requested
/// eigenvector blocks) of one pair of 2-D matrices; both inputs are consumed
/// as scratch (destroyed by `GGEV`). Returns the raw `(alpha, beta)` pair
/// (`lambda = alpha / beta`, see [`write_generalized_eigenvalues`]) with the
/// eigenvectors rescaled to unit 2-norm.
pub fn ref_impl_eig_generalized_simple_f<'b, T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    b: TensorReference<'b, T, B, Ix2>,
    left: bool,
    right: bool,
) -> Result<(
    Tensor<Complex<T::Real>, B, Ix1>,
    Tensor<Complex<T::Real>, B, Ix1>,
    Option<Tensor<Complex<T::Real>, B, Ix2>>,
    Option<Tensor<Complex<T::Real>, B, Ix2>>,
)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let (alpha, beta, vl, vr) =
        device.with_blas_num_threads(nthreads, || GGEV::default().a(a).b(b).left(left).right(right).build()?.run())?;
    let vl = vl.map(normalize_eigvec_columns);
    let vr = vr.map(normalize_eigvec_columns);
    // LAPACK's `beta` is real for a real routine and complex otherwise; the
    // composition passes it on as complex
    let beta: Tensor<Complex<T::Real>, B, Ix1> = rt::asarray((
        beta.raw().iter().map(|&x| Complex::new(x.re(), x.im())).collect::<Vec<_>>(),
        beta.shape().to_vec(),
        &device,
    ))
    .into_dim::<Ix1>();
    Ok((alpha, beta, vl, vr))
}

/// `a / b` in the overflow- and underflow-safe form (as NumPy divides):
/// `num`'s `|b|^2` factor loses the quotient as soon as `|b|` leaves about
/// `1e±154` for `f64` or `1e±19` for `f32`. A real `b` divides component-wise,
/// a complex one by Smith's algorithm.
fn cdiv<R: Float>(a: Complex<R>, b: Complex<R>) -> Complex<R> {
    if b.im.is_zero() {
        Complex::new(a.re / b.re, a.im / b.re)
    } else if b.re.abs() >= b.im.abs() {
        let r = b.im / b.re;
        let d = b.re + r * b.im;
        Complex::new((a.re + a.im * r) / d, (a.im - a.re * r) / d)
    } else {
        let r = b.re / b.im;
        let d = b.im + r * b.re;
        Complex::new((a.re * r + a.im) / d, (a.im * r - a.re) / d)
    }
}

/// Write `w = alpha / beta` into `w` with SciPy's rules for a vanishing
/// `beta`: `alpha != 0` gives (projective) infinity, `alpha = beta = 0` gives
/// `nan` (`nan + 0j` when every `alpha` is real, `nan + nan j` otherwise;
/// decided for this slice — SciPy decides over the whole stack). The quotient
/// is taken in an overflow- and underflow-safe form.
pub fn write_generalized_eigenvalues<R, B>(
    alpha: &[Complex<R>],
    beta: &[Complex<R>],
    w: &mut TensorMut<'_, Complex<R>, B, Ix1>,
) -> Result<()>
where
    R: Float,
    B: DeviceAPI<Complex<R>, Raw = Vec<Complex<R>>>,
{
    assert_eq!(w.shape(), &[alpha.len()]);
    assert_eq!(alpha.len(), beta.len());
    let all_real = alpha.iter().all(|a| a.im == R::zero());
    let offsets: Vec<usize> = IterLayout::new(w.layout(), TensorIterOrder::C)?.collect();
    let raw = w.raw_mut();
    for (k, off) in offsets.into_iter().enumerate() {
        let (a, b) = (alpha[k], beta[k]);
        raw[off] = if b != Complex::<R>::zero() {
            cdiv(a, b)
        } else if a != Complex::<R>::zero() {
            Complex::<R>::new(R::infinity(), R::zero())
        } else if all_real {
            Complex::<R>::new(R::nan(), R::zero())
        } else {
            Complex::<R>::new(R::nan(), R::nan())
        };
    }
    Ok(())
}

/// Write row `r` of a `(..., 2, n)` homogeneous spectrum slice, element `k`
/// taking `f(k)`.
fn write_spectrum_row<R, B>(dst: &mut TensorMut<'_, Complex<R>, B, Ix2>, r: usize, f: impl FnMut(usize) -> Complex<R>)
where
    R: Float,
    B: DeviceAPI<Complex<R>, Raw = Vec<Complex<R>>>,
{
    let layout = dst.layout().clone();
    let n = layout.shape()[1];
    let base = layout.offset() as isize + r as isize * layout.stride()[0];
    let cs = layout.stride()[1];
    let raw = dst.raw_mut();
    let mut f = f;
    for k in 0..n {
        raw[(base + k as isize * cs) as usize] = f(k);
    }
}

/// Allocate the zero-filled `eig` outputs for a stack of `n x n` matrices:
/// the spectrum `(..., n)` or the homogeneous `(..., 2, n)` stack, and the
/// requested eigenvector blocks `(..., n, n)`.
#[allow(clippy::type_complexity)]
fn eig_outputs<T, B>(
    batch_shape: &[usize],
    n: usize,
    left: bool,
    right: bool,
    homogeneous: bool,
    order: FlagOrder,
    device: &B,
) -> Result<(
    Tensor<Complex<T::Real>, B, IxD>,
    Option<Tensor<Complex<T::Real>, B, IxD>>,
    Option<Tensor<Complex<T::Real>, B, IxD>>,
)>
where
    T: BlasFloat,
    B: DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>> + DeviceCreationNumAPI<Complex<T::Real>>,
{
    let w_inner = if homogeneous { vec![2, n] } else { vec![n] };
    let w = zeros_f((stack_shape(batch_shape, &w_inner, order), device))?;
    let block = || zeros_f((stack_shape(batch_shape, &[n, n], order), device));
    let vl = if left { Some(block()?) } else { None };
    let vr = if right { Some(block()?) } else { None };
    Ok((w, vl, vr))
}

/// The static-dim `eig`/`eigvals` overloads type the result by `a`: a `b`
/// with more batch axes than `a` has no static result rank and is rejected
/// there (`static_ndim_a` is `Some`), while the dynamic-dim overloads
/// broadcast instead (`static_ndim_a` is `None`).
pub fn check_static_generalized_ranks(
    ndim_a: usize,
    ndim_b: usize,
    static_ndim_a: Option<usize>,
    entry: &str,
) -> Result<()> {
    rstsr_assert!(
        static_ndim_a.is_none() || ndim_a >= ndim_b,
        InvalidLayout,
        "{entry}: the static-dim overloads type the result by `a`, so `b` cannot have more batch axes than `a` (a is {ndim_a}-D, b is {ndim_b}-D); use dynamic dims"
    )
}

/// n-dim `eig`: the eigenvalues and the requested eigenvector blocks of every
/// square matrix slice of the stack (the two matrix axes are the last two
/// shape axes under a row-major device, the first two under a column-major
/// one; the rest is the batch). With a second matrix `b` the problem is the
/// generalized `A v = lambda B v`, and the batch dims of `a` and `b` broadcast
/// against each other (aligned from the left under a column-major device).
///
/// `left`/`right` select the eigenvector blocks; `homogeneous` gives the
/// `(alpha, beta)` spectrum stacked on the second-to-last axis `(..., 2, n)`
/// (`beta` identically one for the standard problem) instead of the
/// eigenvalues `(..., n)` (the quotient is taken in an overflow- and
/// underflow-safe form; see [`write_generalized_eigenvalues`]).
///
/// The inputs are consumed as scratch: an in-order owned or mutable input is
/// destroyed by the per-slice driver call when its batch needs no broadcast,
/// and nothing is written back; all other inputs are copied into the call
/// order. `entry` names the calling entry (`"eig"` or `"eigvals"`) in the
/// error messages.
#[allow(clippy::type_complexity)]
pub fn ref_impl_eig_nd_f<T, B>(
    a: TensorReference<'_, T, B, IxD>,
    b: Option<TensorReference<'_, T, B, IxD>>,
    side: EigSide,
    homogeneous: bool,
    entry: &str,
) -> Result<(
    Tensor<Complex<T::Real>, B, IxD>,
    Option<Tensor<Complex<T::Real>, B, IxD>>,
    Option<Tensor<Complex<T::Real>, B, IxD>>,
)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>
        + DeviceAPI<Complex<T::Real>, Raw = Vec<Complex<T::Real>>>
        + DeviceCreationAnyAPI<Complex<T::Real>>
        + DeviceCreationNumAPI<Complex<T::Real>>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix1, Ix1>
        + OpAssignAPI<Complex<T::Real>, Ix1>
        + OpAssignArbitaryAPI<Complex<T::Real>, Ix2, Ix2>
        + OpAssignAPI<Complex<T::Real>, Ix2>,
{
    let (left, right) = (side.wants_left(), side.wants_right());
    let device = a.device().clone();
    let order = device.default_order();

    // the inputs go through `overwritable_convert` once
    let mut converted_a = overwritable_convert_with_order(a, order)?;
    let a_stack = StackLayout::new(converted_a.view().layout(), 2, order)?;
    let inner_a = a_stack.inner_shape();
    rstsr_assert_eq!(
        inner_a[0],
        inner_a[1],
        InvalidLayout,
        "{entry}: matrix a must be square, got {}x{}",
        inner_a[0],
        inner_a[1]
    )?;
    let mut converted_b = match b {
        Some(b) => {
            rstsr_assert!(device.same_device(b.device()), DeviceMismatch, "{entry}: a and b are on different devices")?;
            let converted_b = overwritable_convert_with_order(b, order)?;
            let inner_b = StackLayout::new(converted_b.view().layout(), 2, order)?.inner_shape().to_vec();
            rstsr_assert_eq!(
                inner_a,
                inner_b.as_slice(),
                InvalidLayout,
                "{entry}: a and b must share the matrix shape, got {:?} vs {:?}",
                inner_a,
                inner_b
            )?;
            Some(converted_b)
        },
        None => None,
    };
    let b_stack = converted_b.as_ref().map(|cb| StackLayout::new(cb.view().layout(), 2, order)).transpose()?;
    let batch_out = match &b_stack {
        Some(stack) => broadcast_shapes_f(&[a_stack.batch_shape().to_vec(), stack.batch_shape().to_vec()], order)?,
        None => a_stack.batch_shape().to_vec(),
    };
    let n = inner_a[0];
    let (mut w, mut vl, mut vr) = eig_outputs::<T, B>(&batch_out, n, left, right, homogeneous, order, &device)?;
    let n_batch = batch_out.iter().product::<usize>();
    // empty matrix or empty batch: zero-filled outputs, the drivers are not called
    if n == 0 || n_batch == 0 {
        return Ok((w, vl, vr));
    }
    let w_stack = StackLayout::new(w.layout(), if homogeneous { 2 } else { 1 }, order)?;
    let vl_stack = vl.as_ref().map(|v| StackLayout::new(v.layout(), 2, order)).transpose()?;
    let vr_stack = vr.as_ref().map(|v| StackLayout::new(v.layout(), 2, order)).transpose()?;

    // one batch element: the driver call and the output slices
    let mut kernel =
        |a: TensorReference<'_, T, B, Ix2>, b: Option<TensorReference<'_, T, B, Ix2>>, i: usize| -> Result<()> {
            let (alpha, beta, vl_i, vr_i) = match b {
                Some(b) => {
                    let (alpha, beta, vl_i, vr_i) = ref_impl_eig_generalized_simple_f(a, b, left, right)?;
                    (alpha, Some(beta), vl_i, vr_i)
                },
                None => {
                    let (w_i, vl_i, vr_i) = ref_impl_eig_simple_f(a, left, right)?;
                    (w_i, None, vl_i, vr_i)
                },
            };
            // spectrum: (alpha, beta) stack, or eigenvalues (beta = 1 for GEEV)
            match beta {
                Some(beta) if homogeneous => {
                    let mut w_slice = stack_slice_at_mut(w.view_mut(), &w_stack, i)?.into_dim::<Ix2>();
                    write_spectrum_row(&mut w_slice, 0, |k| alpha.raw()[k]);
                    write_spectrum_row(&mut w_slice, 1, |k| beta.raw()[k]);
                },
                Some(beta) => {
                    let mut w_slice = stack_slice_at_mut(w.view_mut(), &w_stack, i)?.into_dim::<Ix1>();
                    write_generalized_eigenvalues(alpha.raw(), beta.raw(), &mut w_slice)?;
                },
                None if homogeneous => {
                    let mut w_slice = stack_slice_at_mut(w.view_mut(), &w_stack, i)?.into_dim::<Ix2>();
                    write_spectrum_row(&mut w_slice, 0, |k| alpha.raw()[k]);
                    write_spectrum_row(&mut w_slice, 1, |_| Complex::one());
                },
                None => stack_slice_at_mut(w.view_mut(), &w_stack, i)?.into_dim::<Ix1>().assign_f(alpha)?,
            }
            if let (Some(vl), Some(vl_i)) = (vl.as_mut(), vl_i) {
                stack_slice_at_mut(vl.view_mut(), vl_stack.as_ref().unwrap(), i)?.into_dim::<Ix2>().assign_f(vl_i)?;
            }
            if let (Some(vr), Some(vr_i)) = (vr.as_mut(), vr_i) {
                stack_slice_at_mut(vr.view_mut(), vr_stack.as_ref().unwrap(), i)?.into_dim::<Ix2>().assign_f(vr_i)?;
            }
            Ok(())
        };

    // an input is destroyed in place only when each of its slices is used once
    // (its batch is the output batch); a slice of a broadcast input is copied
    // out of a view broadcast to the output batch
    let a_in_place = a_stack.batch_shape() == batch_out.as_slice();
    let b_in_place = b_stack.as_ref().is_none_or(|stack| stack.batch_shape() == batch_out.as_slice());
    let full = stack_shape(&batch_out, &[n, n], order);
    for i in 0..n_batch {
        // a slice copied out of a broadcast view lives in these bindings
        #[allow(unused_assignments)]
        let (mut a_copy, mut b_copy) = (None::<Tensor<T, B, Ix2>>, None);
        let a_slice: TensorReference<'_, T, B, Ix2> = if a_in_place {
            stack_slice_at_mut(converted_a.view_mut(), &a_stack, i)?.into_dim::<Ix2>().into()
        } else {
            let a_view = converted_a.view();
            let a_full = a_view.broadcast_to(full.clone());
            let a_full_stack = StackLayout::new(a_full.layout(), 2, order)?;
            a_copy = Some(stack_slice_at(a_full.view(), &a_full_stack, i)?.into_dim::<Ix2>().into_owned());
            a_copy.as_mut().unwrap().view_mut().into()
        };
        let b_slice: Option<TensorReference<'_, T, B, Ix2>> = match converted_b.as_mut() {
            Some(converted_b) => Some(if b_in_place {
                stack_slice_at_mut(converted_b.view_mut(), b_stack.as_ref().unwrap(), i)?.into_dim::<Ix2>().into()
            } else {
                let b_view = converted_b.view();
                let b_full = b_view.broadcast_to(full.clone());
                let b_full_stack = StackLayout::new(b_full.layout(), 2, order)?;
                b_copy = Some(stack_slice_at(b_full.view(), &b_full_stack, i)?.into_dim::<Ix2>().into_owned());
                b_copy.as_mut().unwrap().view_mut().into()
            }),
            None => None,
        };
        kernel(a_slice, b_slice, i)?;
    }
    Ok((w, vl, vr))
}

/* #endregion */

/* #region qr */

/// Generate `Q` in place from the `GEQRF` output: the `k` reflectors in `tau`
/// give the first `k` columns; further columns of `a` are an orthonormal
/// completion.
fn orgqr_2d<'a, 'tau, T, B>(
    a: TensorReference<'a, T, B, Ix2>,
    tau: TensorView<'tau, T, B, Ix1>,
    k: usize,
) -> Result<TensorMutable2<'a, T, B>>
where
    T: BlasFloat,
    B: BlasDriverBaseAPI<T> + ORGQRDriverAPI<T> + OpAssignArbitaryAPI<T, Ix1, Ix1> + OpAssignAPI<T, Ix1>,
{
    ORGQR::default().a(a).tau(tau).k(k).build()?.run()
}

/// QR factorization of a single 2-D matrix (no parallel-mode handling).
///
/// The input is consumed as scratch (destroyed by `GEQRF`). `R` is copied out
/// before `Q` is generated; with `k = min(M, N)`, `Q` is `(M, K)`/`R` is
/// `(K, N)` for [`QrMode::Reduced`], and `(M, M)`/`(M, N)` for
/// [`QrMode::Complete`].
pub fn ref_impl_qr_simple_f<T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    mode: QrMode,
) -> Result<(Tensor<T, B, Ix2>, Tensor<T, B, Ix2>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T> + OpAssignArbitaryAPI<T, Ix1, Ix1> + OpAssignAPI<T, Ix1>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    device.with_blas_num_threads(nthreads, || {
        let [m, n] = *a.shape();
        let k = Ord::min(m, n);
        let (q_cols, r_rows) = match mode {
            QrMode::Reduced => (k, k),
            QrMode::Complete => (m, m),
        };

        // zero-size: Q is the identity completion when only columns are missing
        if k == 0 {
            let q = if q_cols == 0 { zeros_f(([m, 0], &device))? } else { eye_f((m, &device))? }.into_dim::<Ix2>();
            let r = zeros_f(([r_rows, n], &device))?.into_dim::<Ix2>();
            return Ok((q, r));
        }

        let (mut h, tau) = GEQRF::default().a(a).build()?.run()?;
        // R: upper triangle of the first k rows (zero rows below for complete mode)
        let mut r: Tensor<T, B, Ix2> = zeros_f(([r_rows, n], &device))?.into_dim::<Ix2>();
        for i in 0..k {
            r.i_mut((i, i..n)).assign_f(h.view().i((i, i..n)))?;
        }

        // Q: in place on h when it already has q_cols columns; otherwise a
        // fresh buffer holding the leading columns of h (zero-padded completion)
        let q: Tensor<T, B, Ix2> = if q_cols == n {
            orgqr_2d(h.view_mut().into(), tau.view(), q_cols)?.into_owned().into_dim::<Ix2>()
        } else if q_cols < n {
            orgqr_2d(h.view().i((.., 0..q_cols)).into_dim::<Ix2>().into(), tau.view(), q_cols)?
                .into_owned()
                .into_dim::<Ix2>()
        } else {
            // complete mode, m > n: only k reflectors exist; the extra columns
            // are the orthonormal completion generated by ORGQR
            let mut q: Tensor<T, B, Ix2> = zeros_f(([m, q_cols], &device))?.into_dim::<Ix2>();
            q.i_mut((.., 0..n)).assign_f(h.view())?;
            orgqr_2d(q.view_mut().into(), tau.view(), k)?;
            q
        };
        Ok((q, r))
    })
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

/// n-dim `slogdet` over the batch dims (row-major: matrix = last two; col-major: first two).
pub fn ref_impl_slogdet_nd_f<T, B>(a: TensorView<'_, T, B, IxD>) -> Result<(Tensor<T, B, IxD>, Tensor<T::Real, B, IxD>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + DeviceCreationAnyAPI<T::Real>,
{
    use crate::stack::{stack_slice_at, StackLayout};

    let device = a.device().clone();
    let order = device.default_order();
    let stack = StackLayout::new(a.layout(), 2, order)?;
    let inner_shape = stack.inner_shape();
    rstsr_assert_eq!(
        inner_shape[0],
        inner_shape[1],
        InvalidLayout,
        "linalg: expected square matrices, got {}x{}",
        inner_shape[0],
        inner_shape[1]
    )?;

    let batch_shape = stack.batch_shape().to_vec();
    let mut sign: Tensor<T, B, IxD> = zeros_f((batch_shape.clone(), &device))?;
    let mut logabsdet: Tensor<T::Real, B, IxD> = zeros_f((batch_shape, &device))?;
    let mut sign_offsets = IterLayout::new(sign.layout(), TensorIterOrder::C)?;
    let mut log_offsets = IterLayout::new(logabsdet.layout(), TensorIterOrder::C)?;

    for i in 0..stack.len() {
        let slice = stack_slice_at(a.view(), &stack, i)?.into_dim::<Ix2>();
        let (s, l) = ref_impl_slogdet_f(slice.into())?;
        sign.raw_mut()[sign_offsets.next().unwrap_or(i)] = s;
        logabsdet.raw_mut()[log_offsets.next().unwrap_or(i)] = l;
    }
    Ok((sign, logabsdet))
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

/* #region lstsq */

/// Single-slice GELSD under the device BLAS thread policy (same pattern as
/// the `svd` entries of this module).
pub fn ref_impl_lstsq_f<'a, T, B>(
    lstsq_args: GELSD_<'a, B, T>,
) -> Result<(Tensor<T, B, Ix2>, Tensor<T::Real, B, Ix1>, usize, Option<Tensor<T::Real, B, Ix1>>)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let GELSD_ { a, b, rcond } = lstsq_args;
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    device.with_blas_num_threads(nthreads, || GELSD::default().a(a).b(b).rcond(rcond).build()?.run())
}

/* #endregion */

/* #region lu */

/// Row indices of the permutation `A = L[P] @ U` from the 0-based `?GETRF`
/// pivots of a matrix with `m` rows: the pivot swaps applied in reverse
/// order to `0..m`.
pub fn lu_ipiv_to_perm(ipiv: &[usize], m: usize) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..m).collect();
    for i in (0..ipiv.len()).rev() {
        perm.swap(i, ipiv[i]);
    }
    perm
}

/// The inverse of [`lu_ipiv_to_perm`]: the 0-based `?GETRF` pivot sequence
/// that produces the row indices `p` of `A = L[P] @ U`.
pub fn lu_perm_to_ipiv(p: &[usize]) -> Vec<usize> {
    let n = p.len();
    let mut arr = p.to_vec();
    // position of every row in `arr`
    let mut pos = vec![0usize; n];
    for (i, &v) in arr.iter().enumerate() {
        pos[v] = i;
    }
    let mut ipiv = vec![0usize; n];
    for i in 0..n {
        let j = pos[i];
        ipiv[i] = j;
        arr.swap(i, j);
        pos[arr[i]] = i;
        pos[arr[j]] = j;
    }
    ipiv
}

/// LU factorization of one 2-D slice under the device BLAS thread policy:
/// the packed factors, the 0-based pivots as `usize`, and LAPACK's `info`
/// (`info = i > 0`: `U(i, i)` is exactly zero; the factors are complete).
pub fn ref_impl_lu_factor_f<'a, T, B>(
    a: TensorReference<'a, T, B, Ix2>,
) -> Result<(TensorMutable<'a, T, B, Ix2>, Vec<usize>, usize)>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || GETRF::default().a(a).build()?.run_with_info();
    let (lu, ipiv, info) = device.with_blas_num_threads(nthreads, task)?;
    let ipiv = ipiv.raw().iter().map(|&v| v as usize).collect();
    let info = usize::try_from(info).unwrap_or(0);
    Ok((lu, ipiv, info))
}

/// `lu` factors of one slice, written into pre-zeroed buffers: factor the
/// scratch `a` in place, store the row indices in `p: (M,)`, `U` in
/// `u: (K, N)` and `L` in `l: (M, K)` — or the permuted `P L` when
/// `permute_l` is set.
pub fn ref_impl_lu_into_f<T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    mut p: TensorMut<'_, usize, B, Ix1>,
    mut l: TensorMut<'_, T, B, Ix2>,
    mut u: TensorMut<'_, T, B, Ix2>,
    permute_l: bool,
) -> Result<()>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>
        + DeviceAPI<usize, Raw = Vec<usize>>
        + DeviceCreationAnyAPI<T>
        + OpAssignArbitaryAPI<T, IxD, IxD>
        + OpAssignAPI<T, IxD>,
{
    let (lu, ipiv, _info) = ref_impl_lu_factor_f(a)?;
    let perm = lu_ipiv_to_perm(&ipiv, l.shape()[0]);
    let offsets: Vec<usize> = IterLayout::new(p.layout(), TensorIterOrder::C)?.collect();
    let raw = p.raw_mut();
    for (dst, &src) in offsets.iter().zip(perm.iter()) {
        raw[*dst] = src;
    }

    let [m, k] = *l.shape();
    let n = u.shape()[1];
    let lu = lu.view();
    for i in 0..k {
        u.i_mut((i, i..n)).assign_f(lu.i((i, i..n)))?;
    }
    if permute_l {
        for i in 0..m {
            let r = perm[i];
            let c = r.min(k);
            if c > 0 {
                l.i_mut((i, 0..c)).assign_f(lu.i((r, 0..c)))?;
            }
            if r < k {
                l[[i, r]] = T::one();
            }
        }
    } else {
        for j in 0..k {
            l.i_mut((j + 1..m, j)).assign_f(lu.i((j + 1..m, j)))?;
        }
        for i in 0..k {
            l[[i, i]] = T::one();
        }
    }
    Ok(())
}

/// Apply the 0-based `?GETRF` pivots to the rows of `b` in place (`?LASWP`:
/// forward through the pivots for `A x = b`, backward for `Aᵀ x = b`).
pub fn lu_swap_rows<T, B>(mut b: TensorViewMut<'_, T, B, Ix2>, ipiv: &[usize], forward: bool)
where
    T: Clone,
    B: DeviceAPI<T, Raw = Vec<T>>,
{
    let (s0, s1) = (b.layout().stride()[0] as usize, b.layout().stride()[1] as usize);
    let offset = b.layout().offset();
    let nrhs = b.shape()[1];
    let raw = b.raw_mut();
    for i in 0..ipiv.len() {
        let i = if forward { i } else { ipiv.len() - 1 - i };
        let p = ipiv[i];
        if p == i {
            continue;
        }
        for j in 0..nrhs {
            raw.swap(offset + i * s0 + j * s1, offset + p * s0 + j * s1);
        }
    }
}

/// The row swaps and the two triangular solves of one `lu_solve`, in the
/// storage of the contiguous `y` (in the device order).
fn lu_solve_in_place<T, B>(
    l: TensorView<'_, T, B, Ix2>,
    u: TensorView<'_, T, B, Ix2>,
    ipiv: &[usize],
    trans: FlagTrans,
    swaps_first: bool,
    mut y: TensorViewMut<'_, T, B, Ix2>,
) -> Result<()>
where
    T: BlasFloat,
    B: LapackDriverAPI<T>,
{
    let device = y.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    if swaps_first {
        lu_swap_rows(y.view_mut(), ipiv, true);
    }
    let mut step = |tri: TensorView<'_, T, B, Ix2>, uplo: FlagUpLo, diag: FlagDiag| -> Result<()> {
        let task = || {
            TRSM::default()
                .a(tri)
                .b(TensorReference::from(y.view_mut()))
                .uplo(uplo)
                .diag(diag)
                .transa(trans)
                .build()?
                .run()
        };
        device.with_blas_num_threads(nthreads, task)?.clone_to_mut();
        Ok(())
    };
    if swaps_first {
        step(l, Lower, Unit)?;
        step(u, Upper, NonUnit)?;
    } else {
        step(u, Upper, NonUnit)?;
        step(l, Lower, Unit)?;
    }
    if !swaps_first {
        lu_swap_rows(y, ipiv, false);
    }
    Ok(())
}

/// Solve from the LU factors of one slice: `l` unit lower triangular and `u`
/// upper triangular (`(N, N)` each, or the two triangles of one packed factor
/// buffer), `ipiv` the 0-based `?GETRF` pivots, `b` the `(N, K)` right-hand
/// side — overwritten in place when mutable and in the device order, or in a
/// contiguous work buffer written back into `b`. `trans` selects `A x = b`,
/// `Aᵀ x = b` or `Aᴴ x = b`; a singular `U` gives `inf` / `NaN`.
pub fn ref_impl_lu_solve_f<'b, T, B>(
    l: TensorView<'_, T, B, Ix2>,
    u: TensorView<'_, T, B, Ix2>,
    ipiv: &[usize],
    b: TensorReference<'b, T, B, Ix2>,
    trans: FlagTrans,
) -> Result<TensorMutable<'b, T, B, Ix2>>
where
    T: BlasFloat,
    B: LapackDriverAPI<T> + DeviceCreationAnyAPI<T> + OpAssignArbitaryAPI<T, Ix2, Ix2> + OpAssignAPI<T, Ix2>,
{
    let device = b.device().clone();
    rstsr_assert!(
        device.same_device(l.device()) && device.same_device(u.device()),
        DeviceMismatch,
        "lu_solve: the factors and b are on different devices"
    )?;
    let swaps_first = match trans {
        NoTrans => true,
        Trans | ConjTrans => false,
        _ => rstsr_invalid!(trans)?,
    };
    if b.view().shape()[0] == 0 || b.view().shape()[1] == 0 {
        return overwritable_convert(b);
    }
    let order = device.default_order();
    let mut y = overwritable_convert_with_order(b, order)?;
    lu_solve_in_place(l, u, ipiv, trans, swaps_first, y.view_mut())?;
    // a mutable off-order b gets the solution written back
    Ok(y.clone_to_mut())
}

/* #endregion */

/* #region ldl */

/// LAPACK `?SYTRF` / `?HETRF` pivot array → the row-swap sequence and the
/// block sizes of `D` (the port of SciPy's `_ldl_sanitize_ipiv`): `swap` is
/// the row-interchange sequence read in block order, `pivots[k]` is the size
/// of the pivot block starting at row `k` (0 marks the second row of a 2x2
/// block). The scan direction is part of the contract because `swap` is read
/// sequentially.
pub fn ldl_sanitize_ipiv(ipiv: &[blas_int], lower: bool) -> Result<(Vec<usize>, Vec<u8>)> {
    let n = ipiv.len();
    let mut swap: Vec<usize> = (0..n).collect();
    let mut pivots = vec![0u8; n];
    let mut skip_2x2 = false;
    // partner-row and block-start offsets; lower scans indices ascending, upper descending
    let (x, y) = if lower { (1isize, 0isize) } else { (-1isize, -1isize) };
    let step = if lower { 1isize } else { -1isize };
    let mut ind = if lower { 0isize } else { n as isize - 1 };
    for _ in 0..n {
        if skip_2x2 {
            skip_2x2 = false;
        } else {
            let k = ind as usize;
            let cur_val = ipiv[k];
            if cur_val > 0 {
                if cur_val as usize != k + 1 {
                    // index value != array value: permutation required
                    swap[k] = swap[cur_val as usize - 1];
                }
                pivots[k] = 1;
            } else if cur_val < 0 && (0..n as isize).contains(&(ind + x)) && cur_val == ipiv[(ind + x) as usize] {
                // first row of a 2x2 block; the interchanged rows are (-cur-1, -cur)
                if -cur_val != (ind + 2) as blas_int {
                    swap[(ind + x) as usize] = swap[(-cur_val) as usize - 1];
                }
                pivots[(ind + y) as usize] = 2;
                skip_2x2 = true;
            } else {
                return Err(rstsr_error!(InvalidValue, "ldl: invalid pivot array returned by the factorization"));
            }
        }
        ind += step;
    }
    Ok((swap, pivots))
}

/// SciPy `ldl` of one 2-D slice, written into the pre-zeroed `lu` and `d`
/// (`N`, `N`) and `perm` (`N,`): the compact `?SYTRF` / `?HETRF` output is
/// expanded to the explicit block-diagonal `d` and the permuted unit
/// triangular `lu`, with `A = lu @ d @ luᴴ` (or `luᵀ` for the symmetric
/// form) and `lu[perm, :]` triangular. Returns LAPACK's `info` (`i > 0`:
/// `D(i,i)` is exactly zero, the factors are complete).
#[allow(clippy::too_many_arguments)]
pub fn ref_impl_ldl_into_f<T, B>(
    a: TensorReference<'_, T, B, Ix2>,
    hermitian: bool,
    uplo: FlagUpLo,
    mut lu: TensorMut<'_, T, B, Ix2>,
    mut d: TensorMut<'_, T, B, Ix2>,
    mut perm: TensorMut<'_, usize, B, Ix1>,
) -> Result<usize>
where
    T: BlasFloat,
    B: LapackDriverAPI<T> + DeviceAPI<usize, Raw = Vec<usize>>,
{
    let n = a.shape()[0];
    let device = a.device().clone();
    let nthreads = device.get_current_pool().map_or(1, |pool| pool.current_num_threads());
    let task = || match hermitian {
        // `a` is handed over as-is so an in-order mutable slice is factored
        // in place (a view here would copy and leave the caller's input)
        false => SYTRF::<_, _, false>::default().a(a).uplo(uplo).build()?.run_with_info(),
        true => SYTRF::<_, _, true>::default().a(a).uplo(uplo).build()?.run_with_info(),
    };
    let (ldu, ipiv, info) = device.with_blas_num_threads(nthreads, task)?;
    let lower = uplo == Lower;
    let (swap, pivots) = ldl_sanitize_ipiv(ipiv.raw(), lower)?;
    let ldu_v = ldu.view();

    // diagonal of `d`, unit diagonal of `lu`
    for k in 0..n {
        d[[k, k]] = ldu_v[[k, k]];
        lu[[k, k]] = T::one();
    }
    // strict triangle
    if lower {
        for j in 0..n.saturating_sub(1) {
            lu.i_mut((j + 1..n, j)).assign_f(ldu_v.i((j + 1..n, j)))?;
        }
    } else {
        for i in 0..n.saturating_sub(1) {
            lu.i_mut((i, i + 1..n)).assign_f(ldu_v.i((i, i + 1..n)))?;
        }
    }
    // 2x2 blocks: `d` gets both off-diagonal entries (conjugated for the
    // Hermitian form), `lu` clears the cross entry of the outer factor
    let (x, y) = if lower { (1usize, 0usize) } else { (0usize, 1usize) };
    let mut blk_i = 0;
    for &p in pivots.iter().filter(|&&p| p != 0) {
        let inc = blk_i + p as usize;
        if p == 2 {
            let v = ldu_v[[blk_i + x, blk_i + y]];
            d[[blk_i + x, blk_i + y]] = v;
            d[[blk_i + y, blk_i + x]] = if hermitian { v.conj() } else { v };
            lu[[blk_i + x, blk_i + y]] = T::zero();
        }
        blk_i = inc;
    }
    // row interchanges of the outer factor, and the permutation that
    // restores the triangular order (SciPy `_ldl_construct_tri_factor`)
    let mut perm_v: Vec<usize> = (0..n).collect();
    let inds: Vec<usize> = if lower { (0..n).rev().collect() } else { (0..n).collect() };
    for ind in inds {
        let s_ind = swap[ind];
        if s_ind != ind {
            let (col_s, col_e) = match (lower, pivots[ind]) {
                (true, 0) => (ind - 1, n),
                (true, _) => (ind, n),
                (false, 2) => (0, ind + 2),
                (false, _) => (0, ind + 1),
            };
            let offs_a: Vec<usize> =
                IterLayout::new(lu.i((s_ind, col_s..col_e)).layout(), TensorIterOrder::C)?.collect();
            let offs_b: Vec<usize> = IterLayout::new(lu.i((ind, col_s..col_e)).layout(), TensorIterOrder::C)?.collect();
            let raw = lu.raw_mut();
            for (ka, &oa) in offs_a.iter().enumerate() {
                raw.swap(oa, offs_b[ka]);
            }
            perm_v.swap(s_ind, ind);
        }
    }
    // `perm` = argsort of the applied permutation, written through the output view
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by_key(|&i| perm_v[i]);
    let offsets: Vec<usize> = IterLayout::new(perm.layout(), TensorIterOrder::C)?.collect();
    let raw = perm.raw_mut();
    for (dst, &src) in offsets.iter().zip(idx.iter()) {
        raw[*dst] = src;
    }
    Ok(usize::try_from(info).unwrap_or(0))
}

/* #endregion */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_pivot_helpers_are_inverse() {
        // SciPy `lu_factor` docstring example, made 0-based
        assert_eq!(lu_ipiv_to_perm(&[2, 2, 3, 3], 4), vec![1, 3, 0, 2]);
        assert_eq!(lu_perm_to_ipiv(&[1, 3, 0, 2]), vec![2, 2, 3, 3]);
        // tall matrix: rows K..M keep their place
        assert_eq!(lu_ipiv_to_perm(&[2, 2, 3], 5), vec![1, 3, 0, 2, 4]);
        assert_eq!(lu_perm_to_ipiv(&[1, 3, 0, 2, 4]), vec![2, 2, 3, 3, 4]);
        assert_eq!(lu_ipiv_to_perm(&[], 3), vec![0, 1, 2]);
        assert_eq!(lu_perm_to_ipiv(&[0, 1, 2]), vec![0, 1, 2]);

        // every valid pivot sequence (`ipiv[i] >= i`) round-trips
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |n: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize % n
        };
        for m in [1usize, 2, 3, 5, 8, 13] {
            for n in [1usize, 2, 4, 7] {
                let k = m.min(n);
                let ipiv: Vec<usize> = (0..k).map(|i| i + next(m - i)).collect();
                let perm = lu_ipiv_to_perm(&ipiv, m);
                let mut sorted = perm.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..m).collect::<Vec<_>>());
                // rows K..M are never pivoted: the tail of the sequence is identity
                let round = lu_perm_to_ipiv(&perm);
                assert_eq!(&round[..k], ipiv.as_slice());
                assert!(round[k..].iter().enumerate().all(|(i, &v)| v == k + i));
            }
        }
    }

    #[test]
    fn ldl_sanitize_ipiv_matches_scipy() {
        // ground truth generated with SciPy 1.18.1 `_ldl_sanitize_ipiv`
        let cases: Vec<(bool, Vec<blas_int>, Vec<usize>, Vec<u8>)> = vec![
            (true, vec![1, 2, 3], vec![0, 1, 2], vec![1, 1, 1]),
            (true, vec![3, 2, 3], vec![2, 1, 2], vec![1, 1, 1]),
            (true, vec![-2, -2, 3], vec![0, 1, 2], vec![2, 0, 1]),
            (true, vec![1, -3, -3, 4], vec![0, 1, 2, 3], vec![1, 2, 0, 1]),
            (true, vec![2, 2, -3, -3], vec![1, 1, 2, 2], vec![1, 1, 2, 0]),
            (false, vec![1, 2, 3], vec![0, 1, 2], vec![1, 1, 1]),
            (false, vec![3, 2, 3], vec![2, 1, 2], vec![1, 1, 1]),
            (false, vec![1, -2, -2], vec![0, 1, 2], vec![1, 2, 0]),
            (false, vec![1, -3, -3, 4], vec![0, 2, 2, 3], vec![1, 2, 0, 1]),
            (false, vec![3, 3, 1, 4], vec![0, 0, 0, 3], vec![1, 1, 1, 1]),
        ];
        for (lower, ipiv, swap, pivots) in cases {
            let (swap_got, pivots_got) = ldl_sanitize_ipiv(&ipiv, lower).unwrap();
            assert_eq!(swap_got, swap, "swap, ipiv {ipiv:?}, lower {lower}");
            assert_eq!(pivots_got, pivots, "pivots, ipiv {ipiv:?}, lower {lower}");
        }
        for bad in [vec![1, -2, 3], vec![1, 2, 0], vec![1, 0, 2]] {
            assert!(ldl_sanitize_ipiv(&bad, true).is_err(), "invalid ipiv {bad:?}");
        }
    }
}
