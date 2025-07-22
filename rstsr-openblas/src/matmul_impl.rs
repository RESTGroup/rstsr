#![allow(non_camel_case_types)]

use crate::prelude_dev::*;
use num::complex::Complex;
use num::traits::ConstZero;
use rayon::prelude::*;
use rstsr_core::prelude_dev::uninitialized_vec;
use rstsr_lapack_ffi::cblas;
use std::ffi::c_void;

type c32 = Complex<f32>;
type c64 = Complex<f64>;

use cblas::CBLAS_LAYOUT::CblasColMajor as ColMajor;
use cblas::CBLAS_TRANSPOSE::CblasNoTrans as NoTrans;
use cblas::CBLAS_TRANSPOSE::CblasTrans as Trans;
use cblas::CBLAS_UPLO::CblasUpper as Upper;

/* #region gemm */

#[duplicate_item(
     ty    fn_name                 cblas_wrap       ;
    [f32] [gemm_blas_no_conj_f32] [cblas_sgemm_wrap];
    [f64] [gemm_blas_no_conj_f64] [cblas_dgemm_wrap];
    [c32] [gemm_blas_no_conj_c32] [cblas_cgemm_wrap];
    [c64] [gemm_blas_no_conj_c64] [cblas_zgemm_wrap];
)]
#[allow(clippy::too_many_arguments)]
pub fn fn_name(
    c: &mut [ty],
    lc: &Layout<Ix2>,
    a: &[ty],
    la: &Layout<Ix2>,
    b: &[ty],
    lb: &Layout<Ix2>,
    alpha: ty,
    beta: ty,
    pool: Option<&ThreadPool>,
) -> Result<()> {
    // nthreads is only used for `assign_cpu_rayon`.
    // the threading of openblas should be handled outside this function.

    // check layout of output
    if !lc.f_prefer() {
        // change to f-contig anyway
        // we do not handle conj, so this can be done easily
        if lc.c_prefer() {
            // c-prefer, transpose and run
            return fn_name(c, &lc.reverse_axes(), b, &lb.reverse_axes(), a, &la.reverse_axes(), alpha, beta, pool);
        } else {
            // not c-prefer, allocate new buffer and copy back
            let lc_new = lc.shape().new_f_contig(None);
            let mut c_new = unsafe { uninitialized_vec(lc_new.size())? };
            if beta == <ty>::ZERO {
                fill_cpu_rayon(&mut c_new, &lc_new, <ty>::ZERO, pool)?;
            } else {
                assign_cpu_rayon(&mut c_new, &lc_new, c, lc, pool)?;
            }
            fn_name(&mut c_new, &lc_new, a, la, b, lb, alpha, <ty>::ZERO, pool)?;
            assign_cpu_rayon(c, lc, &c_new, &lc_new, pool)?;
            return Ok(());
        }
    }

    // we assume that the layout is correct
    let sc = lc.shape();
    let sa = la.shape();
    let sb = lb.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sa[1], sb[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sb[1], InvalidLayout)?;

    let m = sc[0];
    let n = sc[1];
    let k = sa[1];

    // handle the special case that k is zero-dimensional
    if k == 0 {
        // if k is zero, the result is a zero matrix
        return fill_cpu_rayon(c, lc, <ty>::ZERO, pool);
    }

    // handle the special case that n/m is zero-dimensional
    if n == 0 || m == 0 {
        // if n or m is zero, the result matrix size is zero, and nothing to do
        return Ok(());
    }

    // determine trans/layout and clone data if necessary
    let mut a_data: Option<Vec<ty>> = None;
    let mut b_data: Option<Vec<ty>> = None;
    let (a_trans, la) = if la.f_prefer() {
        (NoTrans, la.clone())
    } else if la.c_prefer() {
        (Trans, la.reverse_axes())
    } else {
        let len = la.size();
        a_data = unsafe { Some(uninitialized_vec(len)?) };
        let la_data = la.shape().new_f_contig(None);
        assign_cpu_rayon(a_data.as_mut().unwrap(), &la_data, a, la, pool)?;
        (NoTrans, la_data)
    };
    let (b_trans, lb) = if lb.f_prefer() {
        (NoTrans, lb.clone())
    } else if lb.c_prefer() {
        (Trans, lb.reverse_axes())
    } else {
        let len = lb.size();
        b_data = unsafe { Some(uninitialized_vec(len)?) };
        let lb_data = lb.shape().new_f_contig(None);
        assign_cpu_rayon(b_data.as_mut().unwrap(), &lb_data, b, lb, pool)?;
        (NoTrans, lb_data)
    };

    // final configuration
    // shape may be broadcasted for one-dimension case, so make this check
    let lda = if la.shape()[1] != 1 { la.stride()[1] } else { la.shape()[0] as isize };
    let ldb = if lb.shape()[1] != 1 { lb.stride()[1] } else { lb.shape()[0] as isize };
    let ldc = if lc.shape()[1] != 1 { lc.stride()[1] } else { lc.shape()[0] as isize };

    let ptr_c = unsafe { c.as_mut_ptr().add(lc.offset()) };
    let ptr_a =
        if let Some(a_data) = a_data.as_ref() { a_data.as_ptr() } else { unsafe { a.as_ptr().add(la.offset()) } };
    let ptr_b =
        if let Some(b_data) = b_data.as_ref() { b_data.as_ptr() } else { unsafe { b.as_ptr().add(lb.offset()) } };

    // actual computation
    unsafe {
        cblas_wrap(ColMajor, a_trans, b_trans, m, n, k, alpha, ptr_a, lda, ptr_b, ldb, beta, ptr_c, ldc);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_sgemm_wrap(
    order: cblas::CBLAS_LAYOUT,
    a_trans: cblas::CBLAS_TRANSPOSE,
    b_trans: cblas::CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    ptr_a: *const f32,
    lda: isize,
    ptr_b: *const f32,
    ldb: isize,
    beta: f32,
    ptr_c: *mut f32,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_sgemm(
            order as cblas::CBLAS_LAYOUT,
            a_trans as cblas::CBLAS_TRANSPOSE,
            b_trans as cblas::CBLAS_TRANSPOSE,
            m as cblas::blas_int,
            n as cblas::blas_int,
            k as cblas::blas_int,
            alpha,
            ptr_a,
            lda as cblas::blas_int,
            ptr_b,
            ldb as cblas::blas_int,
            beta,
            ptr_c,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_dgemm_wrap(
    order: cblas::CBLAS_LAYOUT,
    a_trans: cblas::CBLAS_TRANSPOSE,
    b_trans: cblas::CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    ptr_a: *const f64,
    lda: isize,
    ptr_b: *const f64,
    ldb: isize,
    beta: f64,
    ptr_c: *mut f64,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_dgemm(
            order as cblas::CBLAS_LAYOUT,
            a_trans as cblas::CBLAS_TRANSPOSE,
            b_trans as cblas::CBLAS_TRANSPOSE,
            m as cblas::blas_int,
            n as cblas::blas_int,
            k as cblas::blas_int,
            alpha,
            ptr_a,
            lda as cblas::blas_int,
            ptr_b,
            ldb as cblas::blas_int,
            beta,
            ptr_c,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_cgemm_wrap(
    order: cblas::CBLAS_LAYOUT,
    a_trans: cblas::CBLAS_TRANSPOSE,
    b_trans: cblas::CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: c32,
    ptr_a: *const c32,
    lda: isize,
    ptr_b: *const c32,
    ldb: isize,
    beta: c32,
    ptr_c: *mut c32,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_cgemm(
            order as cblas::CBLAS_LAYOUT,
            a_trans as cblas::CBLAS_TRANSPOSE,
            b_trans as cblas::CBLAS_TRANSPOSE,
            m as cblas::blas_int,
            n as cblas::blas_int,
            k as cblas::blas_int,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as cblas::blas_int,
            ptr_b as *const c_void,
            ldb as cblas::blas_int,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_zgemm_wrap(
    order: cblas::CBLAS_LAYOUT,
    a_trans: cblas::CBLAS_TRANSPOSE,
    b_trans: cblas::CBLAS_TRANSPOSE,
    m: usize,
    n: usize,
    k: usize,
    alpha: c64,
    ptr_a: *const c64,
    lda: isize,
    ptr_b: *const c64,
    ldb: isize,
    beta: c64,
    ptr_c: *mut c64,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_zgemm(
            order as cblas::CBLAS_LAYOUT,
            a_trans as cblas::CBLAS_TRANSPOSE,
            b_trans as cblas::CBLAS_TRANSPOSE,
            m as cblas::blas_int,
            n as cblas::blas_int,
            k as cblas::blas_int,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as cblas::blas_int,
            ptr_b as *const c_void,
            ldb as cblas::blas_int,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as cblas::blas_int,
        );
    }
}

/* #endregion */

/* #region syrk */

#[duplicate_item(
     ty    fn_name                 cblas_wrap       ;
    [f32] [syrk_blas_no_conj_f32] [cblas_ssyrk_wrap];
    [f64] [syrk_blas_no_conj_f64] [cblas_dsyrk_wrap];
    [c32] [syrk_blas_no_conj_c32] [cblas_csyrk_wrap];
    [c64] [syrk_blas_no_conj_c64] [cblas_zsyrk_wrap];
)]
pub fn fn_name(
    c: &mut [ty],
    lc: &Layout<Ix2>,
    a: &[ty],
    la: &Layout<Ix2>,
    alpha: ty,
    pool: Option<&ThreadPool>,
) -> Result<()> {
    // beta is assumed to be zero, and not passed as argument.

    // check layout of output
    if !lc.f_prefer() {
        // change to f-contig anyway
        // we do not handle conj, so this can be done easily
        if lc.c_prefer() {
            // c-prefer, transpose and run
            return fn_name(c, &lc.reverse_axes(), a, la, alpha, pool);
        } else {
            // not c-prefer, allocate new buffer and copy back
            let lc_new = lc.shape().new_f_contig(None);
            let mut c_new = unsafe { uninitialized_vec(lc_new.size())? };
            fill_cpu_rayon(&mut c_new, &lc_new, <ty>::ZERO, pool)?;
            fn_name(&mut c_new, &lc_new, a, la, alpha, pool)?;
            assign_cpu_rayon(c, lc, &c_new, &lc_new, pool)?;
            return Ok(());
        }
    }

    // we assume that the layout is correct
    let sc = lc.shape();
    let sa = la.shape();
    rstsr_assert_eq!(sc[0], sa[0], InvalidLayout)?;
    rstsr_assert_eq!(sc[1], sc[0], InvalidLayout)?;

    let n = sc[0];
    let k = sa[1];

    // handle the special case that k is zero-dimensional
    if k == 0 {
        // if k is zero, the result is a zero matrix
        return fill_cpu_rayon(c, lc, <ty>::ZERO, pool);
    }

    // handle the special case that n is zero-dimensional
    if n == 0 {
        // if n is zero, the result matrix size is zero, and nothing to do
        return Ok(());
    }

    // determine trans/layout and clone data if necessary
    let mut a_data: Option<Vec<ty>> = None;
    let (a_trans, la) = if la.f_prefer() {
        (NoTrans, la.clone())
    } else if la.c_prefer() {
        (Trans, la.reverse_axes())
    } else {
        let len = la.size();
        a_data = unsafe { Some(uninitialized_vec(len)?) };
        let la_data = la.shape().new_f_contig(None);
        assign_cpu_rayon(a_data.as_mut().unwrap(), &la_data, a, la, pool)?;
        (NoTrans, la_data)
    };

    // final configuration
    // shape may be broadcasted for one-dimension case, so make this check
    let lda = if la.shape()[1] != 1 { la.stride()[1] } else { la.shape()[0] as isize };
    let ldc = if lc.shape()[1] != 1 { lc.stride()[1] } else { lc.shape()[0] as isize };

    let ptr_c = unsafe { c.as_mut_ptr().add(lc.offset()) };
    let ptr_a =
        if let Some(a_data) = a_data.as_ref() { a_data.as_ptr() } else { unsafe { a.as_ptr().add(la.offset()) } };

    // actual computation
    unsafe {
        cblas_wrap(ColMajor, Upper, a_trans, n, k, alpha, ptr_a, lda, <ty>::ZERO, ptr_c, ldc);
    }

    // write back to lower triangle
    let n = sc[0];
    let ldc = lc.stride()[1];
    let offset = lc.offset() as isize;
    let task = || {
        (0..(n as isize)).into_par_iter().for_each(|j| {
            ((j + 1)..(n as isize)).for_each(|i| unsafe {
                let idx_ij = (offset + j * ldc + i) as usize;
                let idx_ji = (offset + i * ldc + j) as usize;
                let c_ptr_ij = c.as_ptr().add(idx_ij) as *mut ty;
                *c_ptr_ij = c[idx_ji];
            });
        });
    };
    pool.map_or_else(task, |pool| pool.install(task));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_ssyrk_wrap(
    order: cblas::CBLAS_LAYOUT,
    uplo: cblas::CBLAS_UPLO,
    a_trans: cblas::CBLAS_TRANSPOSE,
    n: usize,
    k: usize,
    alpha: f32,
    ptr_a: *const f32,
    lda: isize,
    beta: f32,
    ptr_c: *mut f32,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_ssyrk(
            order as cblas::CBLAS_LAYOUT,
            uplo as cblas::CBLAS_UPLO,
            a_trans as cblas::CBLAS_TRANSPOSE,
            n as cblas::blas_int,
            k as cblas::blas_int,
            alpha,
            ptr_a,
            lda as cblas::blas_int,
            beta,
            ptr_c,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_dsyrk_wrap(
    order: cblas::CBLAS_LAYOUT,
    uplo: cblas::CBLAS_UPLO,
    a_trans: cblas::CBLAS_TRANSPOSE,
    n: usize,
    k: usize,
    alpha: f64,
    ptr_a: *const f64,
    lda: isize,
    beta: f64,
    ptr_c: *mut f64,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_dsyrk(
            order as cblas::CBLAS_LAYOUT,
            uplo as cblas::CBLAS_UPLO,
            a_trans as cblas::CBLAS_TRANSPOSE,
            n as cblas::blas_int,
            k as cblas::blas_int,
            alpha,
            ptr_a,
            lda as cblas::blas_int,
            beta,
            ptr_c,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_csyrk_wrap(
    order: cblas::CBLAS_LAYOUT,
    uplo: cblas::CBLAS_UPLO,
    a_trans: cblas::CBLAS_TRANSPOSE,
    n: usize,
    k: usize,
    alpha: c32,
    ptr_a: *const c32,
    lda: isize,
    beta: c32,
    ptr_c: *mut c32,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_csyrk(
            order as cblas::CBLAS_LAYOUT,
            uplo as cblas::CBLAS_UPLO,
            a_trans as cblas::CBLAS_TRANSPOSE,
            n as cblas::blas_int,
            k as cblas::blas_int,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as cblas::blas_int,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as cblas::blas_int,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn cblas_zsyrk_wrap(
    order: cblas::CBLAS_LAYOUT,
    uplo: cblas::CBLAS_UPLO,
    a_trans: cblas::CBLAS_TRANSPOSE,
    n: usize,
    k: usize,
    alpha: c64,
    ptr_a: *const c64,
    lda: isize,
    beta: c64,
    ptr_c: *mut c64,
    ldc: isize,
) {
    unsafe {
        cblas::cblas_zsyrk(
            order as cblas::CBLAS_LAYOUT,
            uplo as cblas::CBLAS_UPLO,
            a_trans as cblas::CBLAS_TRANSPOSE,
            n as cblas::blas_int,
            k as cblas::blas_int,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as cblas::blas_int,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as cblas::blas_int,
        );
    }
}

/* #endregion */

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_f32() {
        let a = vec![1., 2., 3., 4., 5., 6.];
        let b = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
        let mut c = vec![0.0; 16];

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        let pool = Some(&pool);
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{c_tsr:}");
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{c_tsr:}");
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{c_tsr:}");
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![61., 70., 79., 88., 76., 88., 100., 112.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 2.0, 0.0, pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{c_tsr:}");
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = 2 * asarray(vec![61., 70., 79., 88., 76., 88., 100., 112.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));
    }

    #[test]
    fn test_c32() {
        let a = linspace((c32::new(1., 1.), c32::new(6., 6.), 6)).into_vec();
        let b = linspace((c32::new(1., 1.), c32::new(12., 12.), 12)).into_vec();
        let mut c = vec![c32::ZERO; 16];

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        let pool = rayon::ThreadPoolBuilder::new().num_threads(16).build().unwrap();
        let pool = Some(&pool);
        gemm_blas_no_conj_c32(&mut c, &lc, &a, &la, &b, &lb, c32::ONE, c32::ZERO, pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{c_tsr:}");
        println!("{:}", c_tsr.reshape([8]));
    }
}
