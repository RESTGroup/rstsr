#![allow(non_camel_case_types)]

use crate::prelude_dev::*;
use num::complex::Complex;
use num::traits::ConstZero;
use rayon::prelude::*;
use rstsr_openblas_ffi::{cblas, ffi};
use std::ffi::c_void;

type c32 = Complex<f32>;
type c64 = Complex<f64>;

const PARALLEL_SWITCH: usize = 64;

/* #region gemm */

macro_rules! impl_gemm_blas_no_conj {
    ($ty: ty, $fn_name: ident, $cblas_wrap: ident) => {
        pub fn $fn_name(
            c: &mut [$ty],
            lc: &Layout<Ix2>,
            a: &[$ty],
            la: &Layout<Ix2>,
            b: &[$ty],
            lb: &Layout<Ix2>,
            alpha: $ty,
            beta: $ty,
            pool: &rayon::ThreadPool,
        ) -> Result<()> {
            // nthreads is only used for `assign_cpu_rayon`.
            // the threading of openblas should be handled outside this function.

            // check layout of output
            if !lc.f_prefer() {
                // change to f-contig anyway
                // we do not handle conj, so this can be done easily
                if lc.c_prefer() {
                    // c-prefer, transpose and run
                    return $fn_name(
                        c,
                        &lc.reverse_axes(),
                        b,
                        &lb.reverse_axes(),
                        a,
                        &la.reverse_axes(),
                        alpha,
                        beta,
                        pool,
                    );
                } else {
                    // not c-prefer, allocate new buffer and copy back
                    let lc_new = lc.shape().new_f_contig(None);
                    let mut c_new = unsafe {
                        let mut c_vec = Vec::with_capacity(lc_new.size());
                        c_vec.set_len(lc_new.size());
                        c_vec
                    };
                    if beta == <$ty>::ZERO {
                        fill_cpu_rayon(&mut c_new, &lc_new, <$ty>::ZERO, pool)?;
                    } else {
                        assign_cpu_rayon(&mut c_new, &lc_new, c, lc, pool)?;
                    }
                    $fn_name(&mut c_new, &lc_new, a, la, b, lb, alpha, <$ty>::ZERO, pool)?;
                    assign_cpu_rayon(c, lc, &c_new, &lc_new, pool)?;
                    return Ok(());
                }
            }

            // we assume that the layout is correct
            let sc = lc.shape();
            let sa = la.shape();
            let sb = lb.shape();
            debug_assert_eq!(sc[0], sa[0]);
            debug_assert_eq!(sa[1], sb[0]);
            debug_assert_eq!(sc[1], sb[1]);

            let m = sc[0];
            let n = sc[1];
            let k = sa[1];

            // determine trans/layout and clone data if necessary
            let mut a_data: Option<Vec<$ty>> = None;
            let mut b_data: Option<Vec<$ty>> = None;
            let (a_trans, la) = if la.f_prefer() {
                (cblas::NoTrans, la.clone())
            } else if la.c_prefer() {
                (cblas::Trans, la.reverse_axes())
            } else {
                let len = la.size();
                a_data = unsafe {
                    let mut a_vec = Vec::with_capacity(len);
                    a_vec.set_len(len);
                    Some(a_vec)
                };
                let la_data = la.shape().new_f_contig(None);
                assign_cpu_rayon(a_data.as_mut().unwrap(), &la_data, a, &la, pool)?;
                (cblas::NoTrans, la_data)
            };
            let (b_trans, lb) = if lb.f_prefer() {
                (cblas::NoTrans, lb.clone())
            } else if lb.c_prefer() {
                (cblas::Trans, lb.reverse_axes())
            } else {
                let len = lb.size();
                b_data = unsafe {
                    let mut b_vec = Vec::with_capacity(len);
                    b_vec.set_len(len);
                    Some(b_vec)
                };
                let lb_data = lb.shape().new_f_contig(None);
                assign_cpu_rayon(b_data.as_mut().unwrap(), &lb_data, b, &lb, pool)?;
                (cblas::NoTrans, lb_data)
            };

            // final configuration
            // shape may be broadcasted for one-dimension case, so make this check
            let lda = if la.shape()[1] != 1 { la.stride()[1] } else { la.shape()[0] as isize };
            let ldb = if lb.shape()[1] != 1 { lb.stride()[1] } else { lb.shape()[0] as isize };
            let ldc = if lc.shape()[1] != 1 { lc.stride()[1] } else { lc.shape()[0] as isize };

            let ptr_c = unsafe { c.as_mut_ptr().add(lc.offset()) };
            let ptr_a = if let Some(a_data) = a_data.as_ref() {
                a_data.as_ptr()
            } else {
                unsafe { a.as_ptr().add(la.offset()) }
            };
            let ptr_b = if let Some(b_data) = b_data.as_ref() {
                b_data.as_ptr()
            } else {
                unsafe { b.as_ptr().add(lb.offset()) }
            };

            // actual computation
            unsafe {
                $cblas_wrap(
                    cblas::ColMajor,
                    a_trans,
                    b_trans,
                    m,
                    n,
                    k,
                    alpha,
                    ptr_a,
                    lda,
                    ptr_b,
                    ldb,
                    beta,
                    ptr_c,
                    ldc,
                );
            }
            Ok(())
        }
    };
}

impl_gemm_blas_no_conj!(f32, gemm_blas_no_conj_f32, cblas_sgemm_wrap);
impl_gemm_blas_no_conj!(f64, gemm_blas_no_conj_f64, cblas_dgemm_wrap);
impl_gemm_blas_no_conj!(c32, gemm_blas_no_conj_c32, cblas_cgemm_wrap);
impl_gemm_blas_no_conj!(c64, gemm_blas_no_conj_c64, cblas_zgemm_wrap);

unsafe fn cblas_sgemm_wrap(
    order: cblas::CblasLayout,
    a_trans: cblas::CblasTranspose,
    b_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_sgemm(
            order as ffi::cblas::CBLAS_ORDER,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            b_trans as ffi::cblas::CBLAS_TRANSPOSE,
            m as ffi::cblas::blasint,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            alpha,
            ptr_a,
            lda as ffi::cblas::blasint,
            ptr_b,
            ldb as ffi::cblas::blasint,
            beta,
            ptr_c,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_dgemm_wrap(
    order: cblas::CblasLayout,
    a_trans: cblas::CblasTranspose,
    b_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_dgemm(
            order as ffi::cblas::CBLAS_ORDER,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            b_trans as ffi::cblas::CBLAS_TRANSPOSE,
            m as ffi::cblas::blasint,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            alpha,
            ptr_a,
            lda as ffi::cblas::blasint,
            ptr_b,
            ldb as ffi::cblas::blasint,
            beta,
            ptr_c,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_cgemm_wrap(
    order: cblas::CblasLayout,
    a_trans: cblas::CblasTranspose,
    b_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_cgemm(
            order as ffi::cblas::CBLAS_ORDER,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            b_trans as ffi::cblas::CBLAS_TRANSPOSE,
            m as ffi::cblas::blasint,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as ffi::cblas::blasint,
            ptr_b as *const c_void,
            ldb as ffi::cblas::blasint,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_zgemm_wrap(
    order: cblas::CblasLayout,
    a_trans: cblas::CblasTranspose,
    b_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_zgemm(
            order as ffi::cblas::CBLAS_ORDER,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            b_trans as ffi::cblas::CBLAS_TRANSPOSE,
            m as ffi::cblas::blasint,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as ffi::cblas::blasint,
            ptr_b as *const c_void,
            ldb as ffi::cblas::blasint,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as ffi::cblas::blasint,
        );
    }
}

/* #endregion */

/* #region syrk */

macro_rules! impl_syrk_blas_no_conj {
    ($ty: ty, $fn_name: ident, $cblas_wrap: ident) => {
        pub fn $fn_name(
            c: &mut [$ty],
            lc: &Layout<Ix2>,
            a: &[$ty],
            la: &Layout<Ix2>,
            alpha: $ty,
            pool: &rayon::ThreadPool,
        ) -> Result<()> {
            // nthreads is only used for `assign_cpu_rayon`.
            // the threading of openblas should be handled outside this function.
            let nthreads = pool.current_num_threads();

            // beta is assumed to be zero, and not passed as argument.

            // check layout of output
            if !lc.f_prefer() {
                // change to f-contig anyway
                // we do not handle conj, so this can be done easily
                if lc.c_prefer() {
                    // c-prefer, transpose and run
                    return $fn_name(c, &lc.reverse_axes(), a, la, alpha, pool);
                } else {
                    // not c-prefer, allocate new buffer and copy back
                    let lc_new = lc.shape().new_f_contig(None);
                    let mut c_new = unsafe {
                        let mut c_vec = Vec::with_capacity(lc_new.size());
                        c_vec.set_len(lc_new.size());
                        c_vec
                    };
                    fill_cpu_rayon(&mut c_new, &lc_new, <$ty>::ZERO, pool)?;
                    $fn_name(&mut c_new, &lc_new, a, la, alpha, pool)?;
                    assign_cpu_rayon(c, lc, &c_new, &lc_new, pool)?;
                    return Ok(());
                }
            }

            // we assume that the layout is correct
            let sc = lc.shape();
            let sa = la.shape();
            debug_assert_eq!(sc[0], sa[0]);
            debug_assert_eq!(sc[1], sc[0]);

            let n = sc[0];
            let k = sa[1];

            // determine trans/layout and clone data if necessary
            let mut a_data: Option<Vec<$ty>> = None;
            let (a_trans, la) = if la.f_prefer() {
                (cblas::NoTrans, la.clone())
            } else if la.c_prefer() {
                (cblas::Trans, la.reverse_axes())
            } else {
                let len = la.size();
                a_data = unsafe {
                    let mut a_vec = Vec::with_capacity(len);
                    a_vec.set_len(len);
                    Some(a_vec)
                };
                let la_data = la.shape().new_f_contig(None);
                assign_cpu_rayon(a_data.as_mut().unwrap(), &la_data, a, &la, pool)?;
                (cblas::NoTrans, la_data)
            };

            // final configuration
            // shape may be broadcasted for one-dimension case, so make this check
            let lda = if la.shape()[1] != 1 { la.stride()[1] } else { la.shape()[0] as isize };
            let ldc = if lc.shape()[1] != 1 { lc.stride()[1] } else { lc.shape()[0] as isize };

            let ptr_c = unsafe { c.as_mut_ptr().add(lc.offset()) };
            let ptr_a = if let Some(a_data) = a_data.as_ref() {
                a_data.as_ptr()
            } else {
                unsafe { a.as_ptr().add(la.offset()) }
            };

            // actual computation
            unsafe {
                $cblas_wrap(
                    cblas::ColMajor,
                    cblas::Upper,
                    a_trans,
                    n,
                    k,
                    alpha,
                    ptr_a,
                    lda,
                    <$ty>::ZERO,
                    ptr_c,
                    ldc,
                );
            }

            // write back to lower triangle
            let n = sc[0];
            let ldc = lc.stride()[1];
            let offset = lc.offset() as isize;
            if n < PARALLEL_SWITCH || nthreads == 1 {
                // lc is always f-prefer
                for j in 0..(n as isize) {
                    for i in (j + 1)..(n as isize) {
                        c[(offset + j * ldc + i) as usize] = c[(offset + i * ldc + j) as usize];
                    }
                }
            } else {
                pool.install(|| {
                    (0..(n as isize)).into_par_iter().for_each(|j| {
                        ((j + 1)..(n as isize)).for_each(|i| unsafe {
                            let idx_ij = (offset + j * ldc + i) as usize;
                            let idx_ji = (offset + i * ldc + j) as usize;
                            let c_ptr_ij = c.as_ptr().add(idx_ij) as *mut $ty;
                            *c_ptr_ij = c[idx_ji];
                        });
                    });
                });
            }
            Ok(())
        }
    };
}

impl_syrk_blas_no_conj!(f32, syrk_blas_no_conj_f32, cblas_ssyrk_wrap);
impl_syrk_blas_no_conj!(f64, syrk_blas_no_conj_f64, cblas_dsyrk_wrap);
impl_syrk_blas_no_conj!(c32, syrk_blas_no_conj_c32, cblas_csyrk_wrap);
impl_syrk_blas_no_conj!(c64, syrk_blas_no_conj_c64, cblas_zsyrk_wrap);

unsafe fn cblas_ssyrk_wrap(
    order: cblas::CblasLayout,
    uplo: cblas::CblasUplo,
    a_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_ssyrk(
            order as ffi::cblas::CBLAS_ORDER,
            uplo as ffi::cblas::CBLAS_UPLO,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            alpha,
            ptr_a,
            lda as ffi::cblas::blasint,
            beta,
            ptr_c,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_dsyrk_wrap(
    order: cblas::CblasLayout,
    uplo: cblas::CblasUplo,
    a_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_dsyrk(
            order as ffi::cblas::CBLAS_ORDER,
            uplo as ffi::cblas::CBLAS_UPLO,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            alpha,
            ptr_a,
            lda as ffi::cblas::blasint,
            beta,
            ptr_c,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_csyrk_wrap(
    order: cblas::CblasLayout,
    uplo: cblas::CblasUplo,
    a_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_csyrk(
            order as ffi::cblas::CBLAS_ORDER,
            uplo as ffi::cblas::CBLAS_UPLO,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as ffi::cblas::blasint,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as ffi::cblas::blasint,
        );
    }
}

unsafe fn cblas_zsyrk_wrap(
    order: cblas::CblasLayout,
    uplo: cblas::CblasUplo,
    a_trans: cblas::CblasTranspose,
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
        ffi::cblas::cblas_zsyrk(
            order as ffi::cblas::CBLAS_ORDER,
            uplo as ffi::cblas::CBLAS_UPLO,
            a_trans as ffi::cblas::CBLAS_TRANSPOSE,
            n as ffi::cblas::blasint,
            k as ffi::cblas::blasint,
            &alpha as *const _ as *const c_void,
            ptr_a as *const c_void,
            lda as ffi::cblas::blasint,
            &beta as *const _ as *const c_void,
            ptr_c as *mut c_void,
            ldc as ffi::cblas::blasint,
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
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, &pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].c();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, &pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![38., 44., 50., 56., 83., 98., 113., 128.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].c();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 1.0, 0.0, &pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
        let c_ref = asarray(vec![61., 70., 79., 88., 76., 88., 100., 112.]);
        assert!(allclose_f64(&c_tsr.map(|v| *v as f64), &c_ref));

        let la = [2, 3].f();
        let lb = [3, 4].c();
        let lc = [2, 4].f();
        gemm_blas_no_conj_f32(&mut c, &lc, &a, &la, &b, &lb, 2.0, 0.0, &pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
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
        gemm_blas_no_conj_c32(&mut c, &lc, &a, &la, &b, &lb, c32::ONE, c32::ZERO, &pool).unwrap();
        let c_tsr = TensorView::new(asarray(&c).into_raw_parts().0, lc);
        println!("{:}", c_tsr);
        println!("{:}", c_tsr.reshape([8]));
    }
}
