use crate::lapack_ffi;
use crate::DeviceBLAS;
use num::Complex;
use rstsr_blas_traits::prelude::*;
use rstsr_common::prelude_dev::*;

use rstsr_native_impl::prelude_dev::*;

use std::slice::from_raw_parts_mut;

#[duplicate_item(
    T     fn_work      func_    ;
   [f32] [sgetrf_work] [sgetrf_];
   [f64] [dgetrf_work] [dgetrf_];
)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    ipiv: *mut blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&(m as _), &(n as _), a, &(lda as _), ipiv, info);
            *info
        },
        RowMajor => {
            let lda_t = m.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -4;
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a_t.as_mut_ptr(), &(lda_t as _), ipiv, info);
            // info > 0: complete factorization with a zero pivot, written back like LAPACKE
            if *info < 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T     fn_work      func_    ;
   [f32] [sgetrf_work] [sgetrf_];
   [f64] [dgetrf_work] [dgetrf_];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        // no workspace; the LAPACKE high tier is a forward
        let mut info: blas_int = 0;
        fn_work(order, m, n, a, lda, ipiv, &mut info)
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cgetrf_work] [cgetrf_];
   [Complex<f64>] [zgetrf_work] [zgetrf_];
)]
pub unsafe fn fn_work(
    order: FlagOrder,
    m: usize,
    n: usize,
    a: *mut T,
    lda: usize,
    ipiv: *mut blas_int,
    info: *mut blas_int,
) -> blas_int {
    use lapack_ffi::lapack::func_;

    match order {
        ColMajor => {
            func_(&(m as _), &(n as _), a as *mut _, &(lda as _), ipiv, info);
            *info
        },
        RowMajor => {
            let lda_t = m.max(1);
            // Check leading dimension(s); Fortran (reference-LAPACK) argument indices
            if lda < n {
                *info = -4;
                return *info;
            }
            // Allocate memory for temporary array(s)
            let mut a_t: Vec<T> = match uninitialized_vec(lda_t * n.max(1)) {
                Ok(a_t) => a_t,
                Err(_) => return -1011,
            };
            let a_slice = from_raw_parts_mut(a, m * lda);
            let la = Layout::new_unchecked([m, n], [lda as isize, 1], 0);
            let la_t = Layout::new_unchecked([m, n], [1, lda_t as isize], 0);
            // Transpose input matrices
            if orderchange_out_r2c_ix2_cpu_serial(&mut a_t, &la_t, a_slice, &la).is_err() {
                return -1011;
            }
            // Call LAPACK function and adjust info
            func_(&(m as _), &(n as _), a_t.as_mut_ptr() as *mut _, &(lda_t as _), ipiv, info);
            // info > 0: complete factorization with a zero pivot, written back like LAPACKE
            if *info < 0 {
                return *info;
            }
            // Transpose output matrices
            if orderchange_out_c2r_ix2_cpu_serial(a_slice, &la, &a_t, &la_t).is_err() {
                return -1011;
            }
            *info
        },
    }
}

#[duplicate_item(
    T              fn_work      func_    ;
   [Complex<f32>] [cgetrf_work] [cgetrf_];
   [Complex<f64>] [zgetrf_work] [zgetrf_];
)]
impl GETRFDriverAPI<T> for DeviceBLAS {
    unsafe fn driver_getrf(
        order: FlagOrder,
        m: usize,
        n: usize,
        a: *mut T,
        lda: usize,
        ipiv: *mut blas_int,
    ) -> blas_int {
        // no workspace; the LAPACKE high tier is a forward
        let mut info: blas_int = 0;
        fn_work(order, m, n, a, lda, ipiv, &mut info)
    }
}
